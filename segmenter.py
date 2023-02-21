# -*- coding: utf-8 -*-
""" Laplacian segmentation """

# Code source: Brian McFee
# License: ISC

from collections import defaultdict
import numpy as np
import scipy

import sklearn.cluster

import librosa
import librosa.feature
from loguru import logger


class Segmenter:
    BINS_PER_OCTAVE: int = 12 * 3
    NUMBER_OF_OCTAVES: int = 7
    EVEC_SMOOTH: int = 9
    REC_SMOOTH: int = 9
    MAX_TYPES: int = 12
    REC_WIDTH: int = 9

    @classmethod
    def make_beat_sync_features(cls, audio_time_series, sampling_rate):

        logger.info("Separating harmonics..")
        audio_with_harmonics = librosa.effects.harmonic(audio_time_series, margin=8)

        logger.info("Computing CQT...")
        db_scaled_audio: np.ndarray = librosa.amplitude_to_db(
                                        librosa.cqt(y=audio_with_harmonics, sr=sampling_rate,
                                                    bins_per_octave=cls.BINS_PER_OCTAVE,
                                                    n_bins=cls.NUMBER_OF_OCTAVES * cls.BINS_PER_OCTAVE),
                                        ref=np.max)

        logger.info("Tracking beats...")
        tempo, beats = librosa.beat.beat_track(y=audio_time_series, sr=sampling_rate, trim=False)
        db_scaled_audio_synced = librosa.util.sync(db_scaled_audio, beats, aggregate=np.median)

        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                    x_min=0,
                                                                    x_max=db_scaled_audio.shape[1]),
                                            sr=sampling_rate)

        logger.info("Computing MFCCs...")
        mel_frequency_cepstral_coefficients = librosa.feature.mfcc(y=audio_time_series, sr=sampling_rate, n_mfcc=13)
        mel_frequency_cepstral_coefficients_synced = librosa.util.sync(mel_frequency_cepstral_coefficients, beats)

        return db_scaled_audio_synced, mel_frequency_cepstral_coefficients_synced, beat_times

    @classmethod
    def embed_beats(cls, A_rep, A_loc):

        logger.info("Building recurrence graph...")
        R = librosa.segment.recurrence_matrix(A_rep, width=cls.REC_WIDTH,
                                              mode='affinity',
                                              metric='cosine',
                                              sym=True)

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, cls.REC_SMOOTH))

        logger.info("Building local graph...")
        path_distance = np.sum(np.diff(A_loc, axis=1) ** 2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        ##########################################################
        # And compute the balanced combination (Equations 6, 7, 9)
        logger.info("Computing the Laplacian")

        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

        A = mu * Rf + (1 - mu) * R_path

        #####################################################
        # Now let's compute the normalized Laplacian (Eq. 10)
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(cls.EVEC_SMOOTH, 1))

        return evecs

    @classmethod
    def cluster(cls, evecs, Cnorm, k, beat_times):
        X = evecs[:, :k] / Cnorm[:, k - 1:k]

        KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)

        seg_ids = KM.fit_predict(X)

        ###############################################################
        # Locate segment boundaries from the label sequence
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beats 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_times = beat_times[bound_beats]

        # Tack on the end-time
        bound_times = list(np.append(bound_times, beat_times[-1]))

        ivals, labs = [], []
        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ivals.append(interval)
            labs.append(str(label))

        return ivals, labs

    @classmethod
    def _reindex_labels(cls, ref_int, ref_lab, est_int, est_lab):
        # for each estimated label
        #    find the reference label that is maximally overlaps with

        score_map = defaultdict(lambda: 0)

        for r_int, r_lab in zip(ref_int, ref_lab):
            for e_int, e_lab in zip(est_int, est_lab):
                score_map[(e_lab, r_lab)] += max(0, min(e_int[1], r_int[1]) -
                                                 max(e_int[0], r_int[0]))

        r_taken = set()
        e_map = dict()

        hits = [(score_map[k], k) for k in score_map]
        hits = sorted(hits, reverse=True)

        while hits:
            cand_v, (e_lab, r_lab) = hits.pop(0)
            if r_lab in r_taken or e_lab in e_map:
                continue
            e_map[e_lab] = r_lab
            r_taken.add(r_lab)

        # Anything left over is unused
        unused = set(est_lab) - set(ref_lab)

        for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
            e_map[e] = u

        return [e_map[e] for e in est_lab]

    @classmethod
    def reindex(cls, hierarchy):
        new_hier = [hierarchy[0]]
        for i in range(1, len(hierarchy)):
            ints, labs = hierarchy[i]
            labs = cls._reindex_labels(new_hier[i - 1][0], new_hier[i - 1][1], ints, labs)
            new_hier.append((ints, labs))

        return new_hier

    @classmethod
    def segment_file(cls, filename):
        logger.info(f"Loading {filename}")
        y, sr = librosa.load(filename)

        logger.info(f"Extracting features...{filename}")
        Csync, Msync, beat_times = cls.make_beat_sync_features(audio_time_series=y, sampling_rate=sr)

        logger.info(f"Constructing embedding...{filename}")
        embedding = cls.embed_beats(Csync, Msync)

        Cnorm = np.cumsum(embedding ** 2, axis=1) ** 0.5

        logger.info(f"Clustering...{filename}")
        segmentations = []
        for k in range(1, cls.MAX_TYPES):
            print('\tk={}'.format(k))
            segmentations.append(cls.cluster(embedding, Cnorm, k, beat_times))

        logger.info('Done.')
        return cls.reindex(segmentations)
