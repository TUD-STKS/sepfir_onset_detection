import numpy as np
import pyswarms as ps
import pandas as pd
import librosa
import madmom
from joblib import Parallel, delayed, dump
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
import random


thr_list = []


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


def reduction(log_filtered_spec, reductionprocessor):
    # Calculate positive spectrogram differences
    diff_spec = reductionprocessor.process(log_filtered_spec)
    # Calculate ODF by summing up spectral differences
    odf = diff_spec.sum(axis=1)
    return odf


def peak_picking(odf, peakprocessor):
    # Peak-Picking
    onsets = peakprocessor.process(odf)
    return onsets


def evaluation(onsets, annotation_file, evaluation_window, combine, delay):
    annotations = madmom.io.load_events(annotation_file)
    evl = OnsetEvaluation(detections=onsets, annotations=annotations,
                          window=evaluation_window, combine=combine,
                          delay=delay)
    return evl


def cost_function(f, spec_list, audiodir, refdir, frame_rate, pre_max,
                  post_max, pre_avg, post_avg, evaluation_window,
                  combine, delay, pearson_score, shift, freq_filter_length,
                  num_rands, rand_min, rand_max):
    c = Parallel(n_jobs=-1)(delayed(iterate_over_all_files)(
        audiodir=audiodir, refdir=refdir, spec_list=spec_list,
        f=f[particle_count], freq_filter_length=freq_filter_length,
        pearson_score=pearson_score, shift=shift, frame_rate=frame_rate,
        pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg,
        evaluation_window=evaluation_window, combine=combine, delay=delay,
        num_rands=num_rands, rand_min=rand_min, rand_max=rand_max)
                            for particle_count in range(f.shape[0]))
    return c


def iterate_over_all_files(audiodir, refdir, spec_list, f, freq_filter_length,
                           pearson_score, shift, frame_rate, pre_max, post_max,
                           pre_avg, post_avg, evaluation_window, combine,
                           delay, num_rands, rand_min, rand_max):
    str_audiodir = audiodir  # Relative path to directory containing audio data
    str_annodir = refdir  # Relative path to directory containing .onset files
    audiodir = os.fsencode(str_audiodir)

    # Create Filter processor
    filter_proc = create_filter_reduction_processor(
        hrow=np.array(f[:freq_filter_length], dtype=np.float32),
        hcol=np.array(f[freq_filter_length:], dtype=np.float32), pos_diff=True)

    if not pearson_score:
        # Find best threshold
        threshold = optimize_threshold(
            list_of_specs=spec_list, filter_proc=filter_proc, refdir=refdir,
            num_rands=num_rands, rand_min=rand_min, rand_max=rand_max,
            pre_max=pre_max, post_max=post_max, pre_avg=pre_avg,
            post_avg=post_avg, combine=combine, delay=delay,
            frame_rate=frame_rate, evaluation_window=evaluation_window)

    list_evals = []  # List containing all evaluation objects

    for file_count, file in enumerate(os.listdir(audiodir)):
        filename = os.fsdecode(file)
        filename = re.sub('\.flac$', '.onsets', filename)
        annotation_file = os.path.join(str_annodir, filename)
        odf = reduction(spec_list[file_count], reductionprocessor=filter_proc)
        if pearson_score:
            score = calculate_pearson_score(odf=odf,
                                            annotation_file=annotation_file,
                                            shift=shift, fps=frame_rate)
            list_evals.append(score)
        else:
            # Create Peak-Picking-Processor with optimized threshold
            peak_proc = create_peakprocessor(
                threshold=threshold, pre_max=pre_max, post_max=post_max,
                pre_avg=pre_avg, post_avg=post_avg, combine=combine,
                delay=delay, frame_rate=frame_rate)
            # Onset Detection with best threshold
            onsets = peak_picking(odf=odf, peakprocessor=peak_proc)
            eval_obj = evaluation(onsets=onsets,
                                  annotation_file=annotation_file,
                                  evaluation_window=evaluation_window,
                                  combine=combine, delay=delay)
            list_evals.append(eval_obj)
        # Evaluate complete dataset
        if pearson_score:
            score = np.mean(list_evals)
        else:
            score = OnsetMeanEvaluation(list_evals).fmeasure
        c = 1 - score
    return c


def calculate_pearson_score(odf, annotation_file, shift, fps):
    # Calculate Pearson Correlation Score
    odf = odf / np.max(np.abs(odf))
    events = madmom.io.load_events(annotation_file)
    events += shift  # Shift events by certain time (in seconds)
    quantized_events = quantize_events(events=events, fps=fps, length=odf.size)
    ref_onsets = smooth(signal=quantized_events,
                        kernel=np.array([0.5, 1, 0.5]))
    # Calculate score using the Pearson correlation coefficients
    score = np.corrcoef(ref_onsets, odf)[0, 1]
    return score


def optimize_threshold(list_of_specs, filter_proc, refdir, num_rands, rand_min,
                       rand_max, pre_max, post_max, pre_avg, post_avg,
                       combine, delay, frame_rate, evaluation_window):
    thresholds = []
    for i in range(0, num_rands):
        n = round(random.uniform(rand_min, rand_max), 2)
        thresholds.append(n)

    odflist = []  # List containing all ODFs and their corresponding references
    # Create a list of ODFs
    for file_count, ref_file in enumerate(os.listdir(refdir)):
        odf = reduction(list_of_specs[file_count],
                        reductionprocessor=filter_proc)
        odflist.append([odf, refdir+"/"+ref_file])

    best_f = 0
    best_thr = 0
    # Try all thresholds and remember the best one
    for thr in thresholds:
        list_evals = []  # List containing all evaluation objects
        peak_proc = create_peakprocessor(threshold=thr, pre_max=pre_max,
                                         post_max=post_max, pre_avg=pre_avg,
                                         post_avg=post_avg, combine=combine,
                                         delay=delay, frame_rate=frame_rate)
        for odf in odflist:
            onsets = peak_picking(odf=odf[0], peakprocessor=peak_proc)
            eval_obj = evaluation(onsets=onsets, annotation_file=odf[1],
                                  evaluation_window=evaluation_window,
                                  combine=combine, delay=delay)
            list_evals.append(eval_obj)
        mean_eval = OnsetMeanEvaluation(list_evals)
        f_new = mean_eval.fmeasure
        if f_new > best_f:
            best_f = f_new
            best_thr = thr
    # print("Best Threshold: " + str(best_thr))
    thr_list.append(best_thr)
    return best_thr


def generate_start_position(time_filter_length, freq_filter_length, rand_min,
                            rand_max, num_particles):
    init_pos = []

    for i in range(num_particles):
        particle = []
        for j in range(time_filter_length+freq_filter_length):
            particle.append(round(random.uniform(rand_min, rand_max), 2))
        init_pos.append(particle)
    return np.array(init_pos, dtype='float64')


def main(audiodir, refdir, sample_rate, normalize_input, frame_size,
         frame_rate, num_bands, fmin, fmax, norm_filters, pearson_score, shift,
         time_filter_length, freq_filter_length, pre_max, post_max, pre_avg,
         post_avg, combine, delay, evaluation_window, num_rands, rand_min,
         rand_max, num_particles, num_iterations):

    # Instantiate Preprocessors with parser parameters
    preprocessor = create_preprocessor(
        sample_rate=sample_rate, normalize_input=normalize_input,
        frame_size=frame_size, frame_rate=frame_rate, num_bands=num_bands,
        fmin=fmin, fmax=fmax, norm_filters=norm_filters)

    # Preprocess audio files
    str_audiodir = audiodir  # Relative path to directory containing audio data
    str_annodir = refdir  # Relative path to directory containing .onset files
    audiodir = os.fsencode(str_audiodir)
    spec_list = []  # List containing all preprocessed spectrograms
    for file in tqdm(os.listdir(audiodir), desc="Preprocessing files"):
        filename = os.fsdecode(file)
        audiofile = os.path.join(str_audiodir, filename)

        # Create spectrograms:
        spec = preprocessing(audiofile=audiofile, sr=sample_rate,
                             preprocessor=preprocessor)
        spec_list.append(spec)

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.3}

    # Define particle dimension
    dimensions = time_filter_length + freq_filter_length

    # Generate random start position
    init_pos = generate_start_position(time_filter_length=time_filter_length,
                                       freq_filter_length=freq_filter_length,
                                       rand_min=-10, rand_max=10,
                                       num_particles=num_particles)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                        dimensions=dimensions, options=options,
                                        init_pos=init_pos)
    kwargs = {'spec_list': spec_list, 'audiodir': audiodir, 'refdir': refdir,
              'frame_rate': frame_rate, 'pre_max': pre_max,
              'post_max': post_max, 'pre_avg': pre_avg, 'post_avg': post_avg,
              'evaluation_window': evaluation_window, 'combine': combine,
              'delay': delay, 'pearson_score': pearson_score, 'shift': shift,
              'freq_filter_length': freq_filter_length, 'num_rands': num_rands,
              'rand_min': rand_min, 'rand_max': rand_max}

    # Perform optimization
    cost, pos = optimizer.optimize(cost_function, iters=num_iterations,
                                   **kwargs)

    # Slice threshold array into chunks for each iteration
    sep_thr_list = [thr_list[i:i + num_particles]
                    for i in range(0, len(thr_list), num_particles)]

    # Store historical optimization data
    data = {
        "cost": optimizer.cost_history,
        "position": optimizer.pos_history,
        # "best positions": optimizer.swarm.best_pos,
        "thresholds": sep_thr_list
    }
    print(sep_thr_list)
    dump(data, "results/PSO/F1_i50_p100_with_start_values_no3.joblib")
    return 0


if __name__ == "__main__":
    # Parser
    p = argparse.ArgumentParser(
        description='PSO of separable filters for onset detection')

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
    p.add_argument('-norm_filters', '--norm_filters', type=bool, default=False,
                   help='Normalize the filter of the filterbank to area 1')

    # Peak picking and evaluation parameters
    p.add_argument('-pearson_score', '--pearson_score', type=bool,
                   default=False, help='Use Pearson Correlation Score instead '
                                       'of Peak-Picking and F1-Score '
                                       'evaluation')
    p.add_argument('-shift', '--shift', type=float, default=-.009,
                   help='Shift ')
    p.add_argument(
        '-time_filter_length', '--time_filter_length', type=int, default=3,
        help='Length of the filter in time direction (must be odd integer)')
    p.add_argument(
        '-freq_filter_length', '--freq_filter_length', type=int, default=3,
        help='Length of the filter in frequency direction (must be odd int)')
    # p.add_argument('-threshold', '--threshold', type=float, default=1.3,
    #                help='Threshold for peak-picking')
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

    # Threshold optimization parameters
    p.add_argument('-num_rands', '--num_rands', type=int, default=50,
                   help='Number of randomly generated thresholds')
    p.add_argument('-rand_max', '--rand_max', type=float, default=10,
                   help='Maximum threshold')
    p.add_argument('-rand_min', '--rand_min', type=float, default=0.1,
                   help='Minimum threshold')

    # PSO arguments
    p.add_argument('-num_particles', '--num_particles', type=int, default=100,
                   help='Number of particles')
    p.add_argument('-num_iterations', '--num_iterations', type=int, default=1,
                   help='Number of iterations')

    args = vars(p.parse_args())
    main(**args)
