#!/usr/bin/python3

import numpy as np
import librosa
import madmom
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor)
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.evaluation.onsets import (OnsetEvaluation, OnsetMeanEvaluation,
                                      OnsetSumEvaluation)
from madmom.processors import SequentialProcessor
from sepfir_onset_detection.sep_fir_processor import (
    SeparableFilterSpectrogramDifference,
    TwoDimensionalSeparableFilterSpectrogramDifference)

from tqdm import tqdm
import os
import re
from time import time


# Feature extraction parameters
sr = 44100  # Sampling rate
fs = 2048  # Frame size
fps = 200  # Frames per second
window = np.hanning  # Hann-Window for STFT
num_bands = 24  # Number of bands per octave
f_min = 27.5  # Minimum frequency
f_max = 16000  # Maximum frequency
diff_max_bins = 3  # Number of bins used for maximum filter

# Peak picking and evaluation parameters
# Threshold for peak-picking, superflux default: 1.1, sepfir (3x3 diff-mean)
# default: 2.91
threshold = 2.91
pre_max = 0.03  # pre_max seconds past information for moving maximum
post_max = 0.03  # post_max seconds future information for moving maximum
pre_avg = 0.1  # pre_avg seconds past information for moving average
post_avg = 0.07  # post_avg seconds future information for moving average
eval_window = 0.05  # Time window around a reference onset
combine = 0.03  # Only report one onset within combine seconds
delay = 0  # Report the detected onsets delay seconds delayed

total_time = []

# Instantiate Processors with above parameters
signal_proc = SignalProcessor(sample_rate=sr, num_channels=1, norm=True)
frames_proc = FramedSignalProcessor(frame_size=fs, fps=fps)
stft_proc = ShortTimeFourierTransformProcessor(window=window)
fil_spec_proc = FilteredSpectrogramProcessor(
    num_bands=num_bands, fmin=f_min, fmax=f_max, norm_filters=False)
logfil_spec_proc = LogarithmicSpectrogramProcessor()
specDiffProc = SpectrogramDifferenceProcessor(diff_max_bins=diff_max_bins,
                                              positive_diffs=True)
peak_proc = OnsetPeakPickingProcessor(threshold=threshold,
                                      pre_max=pre_max, post_max=post_max,
                                      pre_avg=pre_avg, post_avg=post_avg,
                                      combine=combine, delay=delay,fps=fps)

preprocessor = SequentialProcessor([signal_proc, frames_proc, stft_proc,
                                    fil_spec_proc, logfil_spec_proc])
filter_proc = SeparableFilterSpectrogramDifference(
    hrow=np.array([1/3, 1/3, 1/3], dtype=np.float32),
    hcol=np.array([1.0, 0, -1.0], dtype=np.float32), positive_diffs=True)
filter_proc_2d = TwoDimensionalSeparableFilterSpectrogramDifference(
    hrow=np.array([1/3, 1/3, 1/3], dtype=np.float32),
    hcol=np.array([1.0, 0, -1.0], dtype=np.float32), positive_diffs=True)


def preprocessing(preprocessor, audiofile, sr):
    # Load and normalize audio file
    signal, sr = librosa.load(audiofile, sr=sr, mono=True)

    # Calculate preprocessed spectrogram
    log_filtered_spec = preprocessor.process(signal)
    return log_filtered_spec


def sf_reduction(log_filtered_spec, reductionprocessor):
    t1 = time()
    # Calculate positive spectogram differences
    diff_spec = reductionprocessor.process(log_filtered_spec)
    # Calculate ODF by summing up spectral differences
    odf = diff_spec.sum(axis=1)
    t2 = time()
    t = t2 - t1
    total_time.append(t)
    return odf


def peak_picking(odf, peakprocessor):
    # Peak-Picking
    onsets = peakprocessor.process(odf)
    return onsets


def evaluation(onsets, annotationsfile):
    # Evaluation
    # Load annotations
    annotations = madmom.io.load_events(annotationsfile)
    # annotations += -.009
    evl = OnsetEvaluation(detections=onsets,
                          annotations=annotations,
                          window=eval_window,
                          combine=combine,
                          delay=delay)
    return evl


# Perform Superflux Onset Detection on all given data and print average measure
str_audiodir = 'data/audio'  # Relative path to directory containing audio data
str_annodir = 'data/annotations/onsets'  # Relative path to label data
audiodir = os.fsencode(str_audiodir)
list_evals = [] # List containing all evaluation objects
f = open('log.txt', 'a')
for file in tqdm(os.listdir(audiodir), desc="Progress"):
    filename = os.fsdecode(file)
    audiofile = os.path.join(str_audiodir, filename)
    filename = re.sub('\.flac$', '.onsets', filename)
    annotationsfile = os.path.join(str_annodir, filename)

    # Superflux pipeline:
    spec = preprocessing(audiofile=audiofile, sr=sr, preprocessor=preprocessor)
    # Toggle here for superflux: reductionsprocessor=specDiffProc,
    # for separable filters: =filter_proc, for 2d filters: =filter_proc_2d
    odf = sf_reduction(spec, reductionprocessor=filter_proc)
    onsets = peak_picking(odf=odf, peakprocessor=peak_proc)
    eval_obj = evaluation(onsets=onsets, annotationsfile=annotationsfile)

    # Write evaluation measures to log file
    f.write(re.sub('\.onsets$', '', filename) + ":\n")
    f.write("Precision: " + str(round(eval_obj.precision, 2)) + "\n")
    f.write("Recall:    " + str(round(eval_obj.recall, 2)) + "\n")
    f.write("F-Measure: " + str(round(eval_obj.fmeasure, 2)) + "\n\n")
    list_evals.append(eval_obj)
f.close()

# Evaluation over complete dataset
sum_eval = OnsetSumEvaluation(list_evals)
mean_eval = OnsetMeanEvaluation(list_evals)
print(sum_eval)
print(mean_eval)
print("Total time: " + str(np.sum(total_time)))
