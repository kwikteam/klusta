# -*- coding: utf-8 -*-

"""SpikeDetekt algorithm."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
import logging

import numpy as np
from tqdm import tqdm

from ..utils import (Bunch,
                     get_excerpts,
                     chunk_bounds,
                     data_chunk,
                     _concatenate,
                     )
from ..kwik.mea import (_channels_per_group,
                        _probe_adjacency_list,
                        )
from .detect import Thresholder, compute_threshold, FloodFillDetector
from .filter import Filter
from .pca import PCA
from .store import SpikeDetektStore
from .waveform import WaveformExtractor

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _keep_spikes(samples, bounds):
    """Only keep spikes within the bounds `bounds=(start, end)`."""
    start, end = bounds
    return (start <= samples) & (samples <= end)


def _split_spikes(groups, idx=None, **arrs):
    """Split spike data according to the channel group."""
    # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}
    dtypes = {'spike_samples': np.float64,
              'waveforms': np.float32,
              'masks': np.float32,
              }
    groups = np.asarray(groups)
    if idx is not None:
        assert idx.dtype == np.bool
        n_spikes_chunk = np.sum(idx)
        # First, remove the overlapping bands.
        groups = groups[idx]
        arrs_bis = arrs.copy()
        for key, arr in arrs.items():
            arrs_bis[key] = arr[idx]
            assert len(arrs_bis[key]) == n_spikes_chunk
    # Then, split along the group.
    groups_u = np.unique(groups)
    out = {}
    for group in groups_u:
        i = (groups == group)
        out[group] = {}
        for key, arr in arrs_bis.items():
            out[group][key] = _concat(arr[i], dtypes.get(key, None))
    return out


def _array_list(arrs):
    out = np.empty((len(arrs),), dtype=np.object)
    out[:] = arrs
    return out


def _concat(arr, dtype=None):
    out = np.array([_[...] for _ in arr], dtype=dtype)
    return out


def _cut_traces(traces, interval_samples):
    n_samples, n_channels = traces.shape
    # Take a subset if necessary.
    if interval_samples is not None:
        start, end = interval_samples
    else:
        return traces, 0
    assert 0 <= start < end
    # WARNING: this loads all traces into memory! To fix this properly,
    # we'll have to implement lazy chunking in ConcatenatedTraces.
    size = (end - start) * traces.shape[1] * traces.dtype.itemsize / 1024. ** 3
    if size > 1:
        logger.warn("Loading all traces in memory: this will require %.3f GB "
                    "of RAM! ", size)
        logger.warn("To avoid this, do not specify `--interval`. "
                    "This bug will be fixed later.")
    traces = traces[start:end, ...]
    if start > 0:
        # TODO: add offset to the spike samples...
        raise NotImplementedError("Need to add `start` to the "
                                  "spike samples")
    return traces, start


def _subtract_offsets(samples, offsets):
    """Subtract the recording offsets from spike samples.

    Return the subtracted spike samples and the spike_recordings array.

    """
    if samples is None:
        return None, None
    assert isinstance(samples, (list, np.ndarray))
    samples = np.asarray(samples).copy()
    assert len(offsets) >= 2
    assert offsets[0] == 0
    assert offsets[-1] >= samples[-1]
    n = len(offsets) - 1

    # Find where to insert the offsets in the spike samples.
    ind = np.searchsorted(samples, offsets)
    assert ind[0] == 0
    assert len(ind) == n + 1

    spike_recordings = np.zeros(len(samples), dtype=np.int32)

    # Loop over all recordings.
    for rec in range(n):
        # Start of the current recording.
        start_rec = offsets[rec]
        # Spike indices of the first spikes in every recording.
        i, j = ind[rec], ind[rec + 1]
        # Ensure that the selected spikes belong to the current recording.
        if i < len(samples) - 1:
            assert start_rec <= samples[i]
        # Subtract the current recording offset to the selected spikes.
        samples[i:j] -= start_rec
        # Create the spike_recordings array.
        spike_recordings[i:j] = rec
        assert np.all(samples[i:j] >= 0)
        assert np.all(samples[i:j] <= (offsets[rec + 1] - start_rec))

    return samples, spike_recordings


def _relative_channels(channels, adjacency):
    """Renumber channels from absolute indices to relative indices,
    to match arrays used in the detection code.

    Parameters
    ----------

    channels : dict
        A dict {group: list_of_channels}
    adjacency : dict
        A dict {group: set_of_neighbors}

    """
    ch_out = {}
    adj_out = {}
    mapping = {}
    offset = 0
    for group in channels:
        ch = channels[group]
        n = len(ch)
        ch_out[group] = [i + offset for i in range(n)]
        # From absolute to relative indices.
        mapping.update({c: (i + offset) for i, c in enumerate(ch)})
        offset += n
    # Recreate the adjacency dict by applying the mapping to
    # both the keys and values.
    for c, i in mapping.items():
        adj_out[i] = set(mapping[_] for _ in adjacency.get(c, set())
                         if _ in mapping)
    return ch_out, adj_out


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

class SpikeDetekt(object):
    """Spike detection class.

    Parameters
    ----------

    tempdir : str
        Path to the temporary directory used by the algorithm. It should be
        on a SSD for best performance.
    probe : dict
        The probe dictionary.
    **kwargs : dict
        Spike detection parameters.

    """
    def __init__(self, tempdir=None, probe=None, **kwargs):
        super(SpikeDetekt, self).__init__()
        self._tempdir = tempdir
        # Load a probe.
        if probe is not None:
            kwargs['probe_channels'] = _channels_per_group(probe)
            kwargs['probe_adjacency_list'] = _probe_adjacency_list(probe)
        self._kwargs = kwargs
        self._n_channels_per_group = {
            group: len(channels)
            for group, channels in self._kwargs['probe_channels'].items()
        }
        for group in self._n_channels_per_group:
            logger.info("Found %d live channels in group %d.",
                        self._n_channels_per_group[group], group)
        self._groups = sorted(self._n_channels_per_group)
        self._n_features = self._kwargs['n_features_per_channel']
        before = self._kwargs['extract_s_before']
        after = self._kwargs['extract_s_after']
        self._n_samples_waveforms = before + after

    # Processing objects creation
    # -------------------------------------------------------------------------

    def _create_filter(self):
        rate = self._kwargs['sample_rate']
        low = self._kwargs['filter_low']
        high = self._kwargs['filter_high_factor'] * rate
        order = self._kwargs['filter_butter_order']
        return Filter(rate=rate,
                      low=low,
                      high=high,
                      order=order,
                      )

    def _create_thresholder(self, thresholds=None):
        mode = self._kwargs['detect_spikes']
        return Thresholder(mode=mode, thresholds=thresholds)

    def _create_detector(self):
        graph = self._kwargs['probe_adjacency_list']
        probe_channels = self._kwargs['probe_channels']
        join_size = self._kwargs['connected_component_join_size']
        return FloodFillDetector(probe_adjacency_list=graph,
                                 join_size=join_size,
                                 channels_per_group=probe_channels,
                                 )

    def _create_extractor(self, thresholds):
        before = self._kwargs['extract_s_before']
        after = self._kwargs['extract_s_after']
        weight_power = self._kwargs['weight_power']
        probe_channels = self._kwargs['probe_channels']
        return WaveformExtractor(extract_before=before,
                                 extract_after=after,
                                 weight_power=weight_power,
                                 channels_per_group=probe_channels,
                                 thresholds=thresholds,
                                 )

    def _create_pca(self):
        n_pcs = self._kwargs['n_features_per_channel']
        return PCA(n_pcs=n_pcs)

    # Misc functions
    # -------------------------------------------------------------------------

    def update_params(self, **kwargs):
        self._kwargs.update(kwargs)

    # Processing functions
    # -------------------------------------------------------------------------

    def apply_filter(self, data):
        """Filter the traces."""
        filter = self._create_filter()
        return filter(data).astype(np.float32)

    def find_thresholds(self, traces):
        """Find weak and strong thresholds in filtered traces."""
        rate = self._kwargs['sample_rate']
        n_excerpts = self._kwargs['n_excerpts']
        excerpt_size = int(self._kwargs['excerpt_size_seconds'] * rate)
        single = bool(self._kwargs['use_single_threshold'])
        strong_f = self._kwargs['threshold_strong_std_factor']
        weak_f = self._kwargs['threshold_weak_std_factor']

        logger.info("Finding the thresholds...")
        excerpt = get_excerpts(traces,
                               n_excerpts=n_excerpts,
                               excerpt_size=excerpt_size)
        excerpt_f = self.apply_filter(excerpt)
        thresholds = compute_threshold(excerpt_f,
                                       single_threshold=single,
                                       std_factor=(weak_f, strong_f))
        logger.debug("Thresholds: {}.".format(thresholds))
        return {'weak': thresholds[0],
                'strong': thresholds[1]}

    def detect(self, traces_f, thresholds=None):
        """Detect connected waveform components in filtered traces.

        Parameters
        ----------

        traces_f : array
            An `(n_samples, n_channels)` array with the filtered data.
        thresholds : dict
            The weak and strong thresholds.

        Returns
        -------

        components : list
            A list of `(n, 2)` arrays with `sample, channel` pairs.

        """
        # Threshold the data following the weak and strong thresholds.
        thresholder = self._create_thresholder(thresholds)
        # Transform the filtered data according to the detection mode.
        traces_t = thresholder.transform(traces_f)
        # Compute the threshold crossings.
        weak = thresholder.detect(traces_t, 'weak')
        strong = thresholder.detect(traces_t, 'strong')

        # Find dead channels.
        cpg = self._kwargs['probe_channels']
        live_channels = sorted([item for sublist in cpg.values()
                                for item in sublist])
        n_channels = traces_f.shape[1]
        dead_channels = np.setdiff1d(np.arange(n_channels), live_channels)
        logger.debug("Dead channels: %s.", ', '.join(map(str, dead_channels)))

        # Kill threshold crossings on dead channels.
        weak[:, dead_channels] = 0
        strong[:, dead_channels] = 0

        # Run the detection.
        detector = self._create_detector()
        return detector(weak_crossings=weak,
                        strong_crossings=strong)

    def extract_spikes(self, components, traces_f,
                       thresholds=None, keep_bounds=None, s_start=None):
        """Extract spikes from connected components.

        Returns a split object.

        Parameters
        ----------
        components : list
            List of connected components.
        traces_f : array
            Filtered data.
        thresholds : dict
            The weak and strong thresholds.
        keep_bounds : tuple
            (keep_start, keep_end).
        s_start : 0
            Start of the chunk.

        """
        n_spikes = len(components)
        if n_spikes == 0:
            return {}

        # Transform the filtered data according to the detection mode.
        thresholder = self._create_thresholder()
        traces_t = thresholder.transform(traces_f)
        # Extract all waveforms.
        extractor = self._create_extractor(thresholds)
        groups, samples, waveforms, masks = zip(*[extractor(component,
                                                            data=traces_f,
                                                            data_t=traces_t,
                                                            )
                                                  for component in components])

        # Create the return arrays.
        groups = np.array(groups, dtype=np.int32)
        assert groups.shape == (n_spikes,)
        assert groups.dtype == np.int32

        samples = np.array(samples, dtype=np.float64)
        assert samples.shape == (n_spikes,)
        assert samples.dtype == np.float64

        # These are lists of arrays of various shapes (because of various
        # groups).
        waveforms = _array_list(waveforms)
        assert waveforms.shape == (n_spikes,)
        assert waveforms.dtype == np.object

        masks = _array_list(masks)
        assert masks.dtype == np.object
        assert masks.shape == (n_spikes,)

        # Reorder the spikes.
        idx = np.argsort(samples)
        groups = groups[idx]
        samples = samples[idx]
        waveforms = waveforms[idx]
        masks = masks[idx]

        # Remove spikes in the overlapping bands.
        # WARNING: add s_start to spike_samples, because spike_samples
        # is relative to the start of the chunk.
        # It is important to add s_start and not keep_start, because of
        # edge effects between overlapping chunks.
        s_start = s_start or 0
        (keep_start, keep_end) = keep_bounds
        idx = _keep_spikes(samples + s_start, (keep_start, keep_end))

        # Split the data according to the channel groups.
        split = _split_spikes(groups, idx=idx, spike_samples=samples,
                              waveforms=waveforms, masks=masks)
        # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}

        # Assert that spike samples are increasing.
        for group in split:
            samples = split[group]['spike_samples']
            if samples is not None:
                assert np.all(np.diff(samples) >= 0)

        return split

    def waveform_pcs(self, waveforms, masks):
        """Compute waveform principal components.

        Returns
        -------

        pcs : array
            An `(n_features, n_samples, n_channels)` array.

        """
        pca = self._create_pca()
        if waveforms is None or not len(waveforms):
            return
        assert (waveforms.shape[0], waveforms.shape[2]) == masks.shape
        return pca.fit(waveforms, masks)

    def features(self, waveforms, pcs):
        """Extract features from waveforms.

        Returns
        -------

        features : array
            An `(n_spikes, n_channels, n_features)` array.

        """
        pca = self._create_pca()
        out = pca.transform(waveforms, pcs=pcs)
        assert out.dtype == np.float32
        return out

    # Chunking
    # -------------------------------------------------------------------------

    def iter_chunks(self, n_samples):
        """Iterate over chunks."""
        rate = self._kwargs['sample_rate']
        chunk_size = int(self._kwargs['chunk_size_seconds'] * rate)
        overlap = int(self._kwargs['chunk_overlap_seconds'] * rate)
        for chunk_idx, bounds in enumerate(chunk_bounds(n_samples, chunk_size,
                                                        overlap=overlap)):
            yield Bunch(bounds=bounds,
                        s_start=bounds[0],
                        s_end=bounds[1],
                        keep_start=bounds[2],
                        keep_end=bounds[3],
                        keep_bounds=(bounds[2:4]),
                        key=bounds[2],
                        chunk_idx=chunk_idx,
                        )

    def n_chunks(self, n_samples):
        """Number of chunks."""
        return len(list(self.iter_chunks(n_samples)))

    def chunk_keys(self, n_samples):
        return [chunk.key for chunk in self.iter_chunks(n_samples)]

    # Output data
    # -------------------------------------------------------------------------

    def output_data(self):
        """Bunch of values to be returned by the algorithm."""
        sc = self._store.spike_counts
        chunk_keys = self._store.chunk_keys

        # NOTE: deal with multiple recordings.
        samples = self._store.spike_samples()

        s = {}
        r = {}
        for group in self._groups:
            spikes = _concatenate(samples[group])
            # Check increasing spikes.
            if spikes is not None:
                assert np.all(np.diff(spikes) >= 0)
            s[group], r[group] = _subtract_offsets(spikes,
                                                   self.recording_offsets)

        output = Bunch(groups=self._groups,
                       n_chunks=len(chunk_keys),
                       chunk_keys=chunk_keys,
                       spike_samples=s,  # dict
                       recordings=r,  # dict
                       masks=self._store.masks(),
                       features=self._store.features(),
                       n_channels_per_group=self._n_channels_per_group,
                       n_features_per_channel=self._n_features,
                       spike_counts=sc,
                       n_spikes_total=sc(),
                       n_spikes_per_group={group: sc(group=group)
                                           for group in self._groups},
                       n_spikes_per_chunk={chunk_key: sc(chunk_key=chunk_key)
                                           for chunk_key in chunk_keys},
                       )
        return output

    # Main loop
    # -------------------------------------------------------------------------

    def _iter_spikes(self, n_samples, step_spikes=1, thresholds=None):
        """Iterate over extracted spikes (possibly subset).

        Yield a split dictionary `{group: {'waveforms': ..., ...}}`.

        """
        for chunk in self.iter_chunks(n_samples):

            # Extract a few components.
            components = self._store.load(name='components',
                                          chunk_key=chunk.key)
            if components is None or not len(components):
                yield chunk, {}
                continue

            k = np.clip(step_spikes, 1, len(components))
            components = components[::k]

            # Get the filtered chunk.
            chunk_f = self._store.load(name='filtered',
                                       chunk_key=chunk.key)

            # Extract the spikes from the chunk.
            split = self.extract_spikes(components, chunk_f,
                                        keep_bounds=chunk.keep_bounds,
                                        s_start=chunk.s_start,
                                        thresholds=thresholds,
                                        )

            yield chunk, split

    def step_detect(self, traces=None, thresholds=None):
        n_samples, n_channels = traces.shape
        n_chunks = self.n_chunks(n_samples)

        # Use the recording offsets when dealing with multiple recordings.
        self.recording_offsets = getattr(traces, 'offsets', [0, len(traces)])

        # Pass 1: find the connected components and count the spikes.
        # self._pr.start_step('detect', n_chunks)

        # Dictionary {chunk_key: components}.
        # Every chunk has a unique key: the `keep_start` integer.
        n_spikes_total = 0
        for chunk in tqdm(self.iter_chunks(n_samples),
                          desc='Detecting spikes'.ljust(24),
                          total=n_chunks, leave=True):
            chunk_data = data_chunk(traces, chunk.bounds, with_overlap=True)

            # Apply the filter.
            data_f = self.apply_filter(chunk_data)
            assert data_f.dtype == np.float32
            assert data_f.shape == chunk_data.shape

            # Save the filtered chunk.
            self._store.store(name='filtered', chunk_key=chunk.key,
                              data=data_f)

            # Detect spikes in the filtered chunk.
            components = self.detect(data_f, thresholds=thresholds)
            self._store.store(name='components', chunk_key=chunk.key,
                              data=components)

            # Report progress.
            n_spikes_chunk = len(components)
            n_spikes_total += n_spikes_chunk
            logger.debug("Found %d spikes in chunk %d.", n_spikes_chunk,
                         chunk.key)
            # self._pr.increment(n_spikes=n_spikes_chunk,
            #                    n_spikes_total=n_spikes_total)

        return n_spikes_total

    def step_excerpt(self, n_samples=None,
                     n_spikes_total=None, thresholds=None):
        # self._pr.start_step('excerpt', self.n_chunks(n_samples))
        n_chunks = self.n_chunks(n_samples)
        k = int(n_spikes_total / float(self._kwargs['pca_n_waveforms_max']))
        w_subset = defaultdict(list)
        m_subset = defaultdict(list)
        n_spikes_total = 0
        for chunk, split in tqdm(self._iter_spikes(n_samples, step_spikes=k,
                                                   thresholds=thresholds),
                                 desc='Extracting waveforms'.ljust(24),
                                 total=n_chunks,
                                 leave=True,
                                 ):
            n_spikes_chunk = 0
            for group, out in split.items():
                w_subset[group].append(out['waveforms'])
                m_subset[group].append(out['masks'])
                assert len(out['masks']) == len(out['waveforms'])
                n_spikes_chunk += len(out['masks'])

            n_spikes_total += n_spikes_chunk
            # self._pr.increment(n_spikes=n_spikes_chunk,
            #                    n_spikes_total=n_spikes_total)
        for group in self._groups:
            w_subset[group] = _concatenate(w_subset[group])
            m_subset[group] = _concatenate(m_subset[group])

        return w_subset, m_subset

    def step_pcs(self, w_subset=None, m_subset=None):
        # self._pr.start_step('pca', len(self._groups))
        pcs = {}
        for group in tqdm(self._groups,
                          desc='Performing PCA'.ljust(24),
                          total=len(self._groups),
                          leave=True,
                          ):
            # Perform PCA and return the components.
            pcs[group] = self.waveform_pcs(w_subset[group],
                                           m_subset[group])
            # self._pr.increment()
        return pcs

    def step_extract(self, n_samples=None,
                     pcs=None, thresholds=None):
        # self._pr.start_step('extract', self.n_chunks(n_samples))
        # chunk_counts = defaultdict(dict)  # {group: {key: n_spikes}}.
        # n_spikes_total = 0
        for chunk, split in tqdm(self._iter_spikes(n_samples,
                                                   thresholds=thresholds),
                                 desc='Computing features'.ljust(24),
                                 total=self.n_chunks(n_samples),
                                 leave=True,
                                 ):
            # Delete filtered and components cache files.
            # self._store.delete(name='filtered', chunk_key=chunk.key)
            # self._store.delete(name='components', chunk_key=chunk.key)
            # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}
            for group, out in split.items():
                if pcs[group] is None:
                    continue
                out['features'] = self.features(out['waveforms'], pcs[group])
                # Checking that spikes are increasing.
                spikes = out['spike_samples']
                self._store.append(group=group,
                                   chunk_key=chunk.key,
                                   spike_samples=spikes,
                                   features=out['features'],
                                   masks=out['masks'],
                                   spike_offset=chunk.s_start,
                                   )
            # n_spikes_total = self._store.spike_counts()
            # n_spikes_chunk = self._store.spike_counts(chunk_key=chunk.key)
            # self._pr.increment(n_spikes_total=n_spikes_total,
            #                    n_spikes=n_spikes_chunk)

    def run_serial(self, traces, interval_samples=None):
        """Run SpikeDetekt using one CPU."""
        traces, offset = _cut_traces(traces, interval_samples)
        n_samples, n_channels = traces.shape

        # Initialize the main loop.
        chunk_keys = self.chunk_keys(n_samples)
        # n_chunks = len(chunk_keys)

        # TODO: progress reporter
        class PR(object):
            def start_step(self, *args, **kwargs):
                pass

            def increment(self, *args, **kwargs):
                pass

        # self._pr = PR(n_chunks=n_chunks)
        self._store = SpikeDetektStore(self._tempdir,
                                       groups=self._groups,
                                       chunk_keys=chunk_keys)

        # Find the weak and strong thresholds.
        thresholds = self.find_thresholds(traces)

        # Spike detection.
        n_spikes_total = self.step_detect(traces=traces,
                                          thresholds=thresholds)

        # Excerpt waveforms.
        w_subset, m_subset = self.step_excerpt(n_samples=n_samples,
                                               n_spikes_total=n_spikes_total,
                                               thresholds=thresholds)

        # Compute the PCs.
        pcs = self.step_pcs(w_subset=w_subset, m_subset=m_subset)

        # Compute all features.
        self.step_extract(n_samples=n_samples, pcs=pcs, thresholds=thresholds)

        return self.output_data()
