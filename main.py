"""
Main Code to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause


import numpy as np
import librosa
import madmom
import random
from madmom.utils import quantize_events
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor, smooth
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor)
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.evaluation.onsets import (OnsetEvaluation, OnsetMeanEvaluation,
                                      OnsetSumEvaluation)
from madmom.processors import SequentialProcessor
from sepfir_onset_detection.sep_fir_processor import \
    SeparableFilterSpectrogramDifference
from tqdm import tqdm
import os
import re
import argparse
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns


def main(audiodir, refdir, sample_rate, normalize_input, frame_size,
         frame_rate, num_bands, fmin, fmax, norm_filters, diff_max_bins,
         pearson_score, threshold, pre_max, post_max, pre_avg, post_avg,
         combine, delay, evaluation_window):
    # Instantiate Processors with parser parameters
    preprocessor = create_preprocessor(
        sample_rate=sample_rate, normalize_input=normalize_input,
        frame_size=frame_size, frame_rate=frame_rate, num_bands=num_bands,
        fmin=fmin, fmax=fmax, norm_filters=norm_filters)
    peak_proc = create_peakprocessor(threshold=threshold, pre_max=pre_max,
                                     post_max=post_max, pre_avg=pre_avg,
                                     post_avg=post_avg, combine=combine,
                                     delay=delay, frame_rate=frame_rate)
    spec_diff_proc = create_sf_reduction_processor(diff_max_bins=diff_max_bins)
    # filter_proc = create_filter_reduction_processor(hrow=np.array([1/3, 1/3, 1/3], dtype=np.float32), hcol=np.array([1.0, 0, -1.0], dtype=np.float32), pos_diff=True)
    filter_proc = create_filter_reduction_processor(
        hrow=np.array([3.88863583, 0.98057879, 2.51280282], dtype=np.float32),
        hcol=np.array([1.4762388, 3.91110099, -5.50139982], dtype=np.float32),
        pos_diff=True)

    # Perform Onset Detection on all given data and print average measure
    str_audiodir = audiodir  # Relative path to directory containing audio data
    str_annodir = refdir  # Relative path to directory containing label files
    audiodir = os.fsencode(str_audiodir)
    list_evals = []  # List containing all evaluation objects
    f = open('log.txt', 'a')
    for file in tqdm(os.listdir(audiodir), desc="Progress"):
        filename = os.fsdecode(file)
        audiofile = os.path.join(str_audiodir, filename)
        filename = re.sub('\.flac$', '.onsets', filename)
        annotation_file = os.path.join(str_annodir, filename)

        # Onset Detection Pipeline:
        spec = preprocessing(audiofile=audiofile, sr=sample_rate,
                             preprocessor=preprocessor)
        # odf = reduction(spec, reduction_processor=spec_diff_proc)
        odf = reduction(spec, reduction_processor=filter_proc)

        if pearson_score:
            score = calculate_pearson_score(
                odf=odf, annotation_file=annotation_file, shift=-.009,
                fps=frame_rate)
            f.write(re.sub('\.onsets$', '', filename) + ":\n")
            f.write("Score: " + str(round(score, 2)) + "\n")
            list_evals.append(score)
        else:
            onsets = peak_picking(odf=odf, peak_processor=peak_proc)
            eval_obj = evaluation(
                onsets=onsets, annotation_file=annotation_file,
                evaluation_window=evaluation_window, combine=combine,
                delay=delay)
            # dump(eval_obj, "results/superflux/"+re.sub('\.onsets$','', filename)+".joblib")
            # Write evaluation measures to log file
            f.write(re.sub('\.onsets$', '', filename) + ":\n")
            f.write("Precision: " + str(round(eval_obj.precision, 2)) + "\n")
            f.write("Recall:    " + str(round(eval_obj.recall, 2)) + "\n")
            f.write("F-Measure: " + str(round(eval_obj.fmeasure, 2)) + "\n\n")
            list_evals.append(eval_obj)

    f.close()

    # Evaluate complete dataset
    if pearson_score:
        pearson_mean = np.mean(list_evals)
        print("Overall Pearson-Score (mean): " + str(pearson_mean))
    else:
        sum_eval = OnsetSumEvaluation(list_evals)
        mean_eval = OnsetMeanEvaluation(list_evals)
        print(sum_eval)
        print(mean_eval)
    return 0


def create_preprocessor(sample_rate, normalize_input, frame_size, frame_rate,
                        num_bands, fmin, fmax, norm_filters):
    # Instantiate Processors with parser parameters
    signal_proc = SignalProcessor(sample_rate=sample_rate, num_channels=1,
                                  norm=normalize_input)
    frames_proc = FramedSignalProcessor(frame_size=frame_size, fps=frame_rate)
    stft_proc = ShortTimeFourierTransformProcessor(window=np.hanning)
    fil_spec_proc = FilteredSpectrogramProcessor(num_bands=num_bands,
                                                 fmin=fmin, fmax=fmax,
                                                 norm_filters=norm_filters)
    logfil_spec_proc = LogarithmicSpectrogramProcessor()
    preprocessor = SequentialProcessor([signal_proc, frames_proc, stft_proc,
                                        fil_spec_proc, logfil_spec_proc])
    return preprocessor


def create_peakprocessor(threshold, pre_max, post_max, pre_avg, post_avg,
                         combine, delay, frame_rate):
    peak_proc = OnsetPeakPickingProcessor(threshold=threshold,
                                          pre_max=pre_max, post_max=post_max,
                                          pre_avg=pre_avg, post_avg=post_avg,
                                          combine=combine, delay=delay,
                                          fps=frame_rate)
    return peak_proc


def create_filter_reduction_processor(hrow, hcol, pos_diff):
    filter_proc = SeparableFilterSpectrogramDifference(hrow=hrow, hcol=hcol,
                                                       positive_diffs=pos_diff)
    return filter_proc


def create_sf_reduction_processor(diff_max_bins):
    spec_diff_proc = SpectrogramDifferenceProcessor(
        diff_max_bins=diff_max_bins, positive_diffs=True)
    return spec_diff_proc


def preprocessing(audiofile, sr, preprocessor):
    # Load and normalize audio file
    signal, sr = librosa.load(audiofile, sr=sr, mono=True)

    # Calculate preprocessed spectrogram
    log_filtered_spec = preprocessor.process(signal)
    return log_filtered_spec


def reduction(log_filtered_spec, reduction_processor):
    # Calculate positive spectrogram differences
    diff_spec = reduction_processor.process(log_filtered_spec)
    # Calculate ODF by summing up spectral differences
    odf = diff_spec.sum(axis=1)
    return odf


def peak_picking(odf, peak_processor):
    # Peak-Picking
    onsets = peak_processor.process(odf)
    return onsets


def evaluation(onsets, annotation_file, evaluation_window, combine, delay):
    # Evaluation
    # Load annotations
    annotations = madmom.io.load_events(annotation_file)
    evl = OnsetEvaluation(detections=onsets, annotations=annotations,
                          window=evaluation_window, combine=combine,
                          delay=delay)
    return evl


def calculate_pearson_score(odf, annotation_file, shift, fps):
    # Calculate R2 Score
    odf = odf / np.max(np.abs(odf))
    events = madmom.io.load_events(annotation_file)
    events += shift  # Shift events by certain time (in seconds)
    quantized_events = quantize_events(events=events, fps=fps, length=odf.size)
    ref_onsets = smooth(
        signal=quantized_events, kernel=np.array([0.5, 1, 0.5]))
    # Calculate score using the Pearson correlation coefficients
    score = np.corrcoef(ref_onsets, odf)[0, 1]
    return score


def optimize_threshold(audiodir, refdir, sample_rate, normalize_input,
                       frame_size, frame_rate, num_bands, fmin, fmax,
                       norm_filters, diff_max_bins, pre_max, post_max, pre_avg,
                       post_avg, combine, delay, evaluation_window, rand_min,
                       rand_max, rands):
    # Instantiate Processors with parser parameters
    preprocessor = create_preprocessor(
        sample_rate=sample_rate, normalize_input=normalize_input,
        frame_size=frame_size, frame_rate=frame_rate, num_bands=num_bands,
        fmin=fmin, fmax=fmax, norm_filters=norm_filters)
    specDiffProc = create_sf_reduction_processor(diff_max_bins=diff_max_bins)
    filter_proc = create_filter_reduction_processor(
        hrow=np.array([0.08, 0.88, 1.56], dtype=np.float32),
        hcol=np.array([0.72, 0.12, -0.89], dtype=np.float32), pos_diff=True)
    thresholds = []
    for i in range(0, rands):
        n = round(random.uniform(rand_min, rand_max), 2)
        thresholds.append(n)

    # Uncomment the following block to iterate over all files in the directory!
    # (No separate folds)
    str_audiodir = audiodir  # Relative path to directory containing audio data
    str_annodir = refdir  # Relative path to directory containing .onset files
    audiodir = os.fsencode(str_audiodir)
    odflist = []  # List containing all ODFs and corresponding reference onsets

    # Create a list of ODFs
    for file in tqdm(os.listdir(audiodir), desc="(1/2) Computing ODFs"):
        filename = os.fsdecode(file)
        audiofile = os.path.join(str_audiodir, filename)
        filename = re.sub('\.flac$', '.onsets', filename)
        annotationsfile = os.path.join(str_annodir, filename)
        # ref_onsets = madmom.io.load_events(annotation_file)

        # Onset Detection Pipeline:
        spec = preprocessing(audiofile=audiofile, sr=sample_rate,
                             preprocessor=preprocessor)
        # odf = reduction(spec, reduction_processor=specDiffProc)
        odf = reduction(spec, reduction_processor=filter_proc)
        odflist.append([odf, annotationsfile])

    best_f = 0
    best_thr = 0
    # Try all thresholds
    for thr in tqdm(thresholds, desc="(2/2) Trying thresholds"):
        list_evals = []  # List containing all evaluation objects
        peak_proc = create_peakprocessor(threshold=thr, pre_max=pre_max,
                                         post_max=post_max, pre_avg=pre_avg,
                                         post_avg=post_avg, combine=combine,
                                         delay=delay, frame_rate=frame_rate)
        for odf in odflist:
            onsets = peak_picking(odf=odf[0], peak_processor=peak_proc)
            eval_obj = evaluation(onsets=onsets, annotation_file=odf[1],
                                  evaluation_window=evaluation_window,
                                  combine=combine, delay=delay)
            list_evals.append(eval_obj)
        mean_eval = OnsetMeanEvaluation(list_evals)
        f_new = mean_eval.fmeasure
        if f_new > best_f:
            best_f = f_new
            best_thr = thr
    return best_thr


if __name__ == "__main__":
    # Parser
    p = argparse.ArgumentParser(description='Superflux onsets.')

    # Path parameters
    p.add_argument('-audiodir', '--audiodir', type=str, default='data/audio',
                   help='Path to audio data')
    p.add_argument('-refdir', '--refdir', type=str,
                   default='data/annotations/onsets',
                   help='Path to reference onset annotations')

    # Feature extraction parameters
    p.add_argument('-norm', '--normalize_input', type=bool, default=True,
                   help='Normalize the input audio signal')
    p.add_argument('-frame_size', '--frame_size', type=int, default=2048,
                   help='Frame size')
    p.add_argument('-sr', '--sample_rate', type=int, default=44100,
                   help='Sampling rate')
    p.add_argument('-fps', '--frame_rate', type=int, default=200,
                   help='Frames per second')
    p.add_argument('-num_bands', '--num_bands', type=int, default=24,
                   help='Number of bands per octave')
    p.add_argument('-fmin', '--fmin', type=float, default=27.5,
                   help='Minimum frequency')
    p.add_argument('-fmax', '--fmax', type=float, default=16000,
                   help='Maximum frequency')
    p.add_argument('-diff_max_bins', '--diff_max_bins', type=int, default=3,
                   help='Number of bins used for maximum filter')
    p.add_argument('-norm_filters', '--norm_filters', type=bool, default=False,
                   help='Normalize the filter of the filterbank to area 1')

    # Peak picking and evaluation parameters
    p.add_argument(
        '-pearson_score', '--pearson_score', type=bool, default=False,
        help='Use Pearson-Scoring-Function instead of Peak-Picking and '
             'F1-Score evaluation')
    p.add_argument('-threshold', '--threshold', type=float, default=4.67,
                   help='Threshold for peak-picking. 1.3 good for superflux')
    p.add_argument(
        '-pre_max', '--pre_max', type=float, default=0.03,
        help='Use pre_max seconds past information for moving maximum')
    p.add_argument(
        '-post_max', '--post_max', type=float, default=0.03,
        help='Use post_max seconds future information for moving maximum')
    p.add_argument(
        '-pre_avg', '--pre_avg', type=float, default=0.1,
        help='Use pre_avg seconds past information for moving average')
    p.add_argument(
        '-post_avg', '--post_avg', type=float, default=0.07,
        help='Use post_avg seconds future information for moving average')
    p.add_argument('-eval_window', '--evaluation_window', type=float,
                   default=0.05, help='Time window around a reference onset')
    p.add_argument('-combine', '--combine', type=float, default=0.03,
                   help='Only report one onset within combine seconds')
    p.add_argument('-delay', '--delay', type=float, default=0,
                   help='Report the detected onsets delay seconds delayed')

    args = vars(p.parse_args())

    main(**args)
