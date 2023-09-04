import numpy as np
import pyswarms as ps
import librosa
import madmom
from joblib import Parallel, delayed, dump
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor)
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.evaluation.onsets import OnsetEvaluation, OnsetSumEvaluation
from madmom.processors import SequentialProcessor
from sepfir_onset_detection.sep_fir_processor import \
    SeparableFilterSpectrogramDifference
from tqdm import tqdm
import os
import argparse
import logging


np.random.seed(42)


def create_preprocessor(sample_rate: int = 44100, normalize_input: bool = True,
                        frame_size: int = 2048, frame_rate: int = 200,
                        num_bands: int = 24, fmin: float = 27.5,
                        fmax: float = 16000, norm_filters: bool = False):
    """
    Instantiate the preprocessor for audio signal processing.

    :param sample_rate: int, default=44100
    :param normalize_input: bool, default=True
    :param frame_size: int, default=2048
    :param frame_rate: int, default=200
    :param num_bands: int, default=24
    :param fmin: float, default=27.5
    :param fmax: float, default = 16000
    :param norm_filters: default = False
    :return: preprocessor: SequentialProcessor
    """
    # Instantiate Processors with parser parameters
    signal_proc = SignalProcessor(sample_rate=sample_rate, num_channels=1,
                                  norm=normalize_input)
    frames_proc = FramedSignalProcessor(frame_size=frame_size, fps=frame_rate)
    stft_proc = ShortTimeFourierTransformProcessor(window=np.hanning)
    fil_spec_proc = FilteredSpectrogramProcessor(
        num_bands=num_bands, fmin=fmin, fmax=fmax, norm_filters=norm_filters)
    logfil_spec_proc = LogarithmicSpectrogramProcessor()
    preprocessor = SequentialProcessor([signal_proc, frames_proc, stft_proc,
                                        fil_spec_proc, logfil_spec_proc])
    return preprocessor


def create_peakprocessor(threshold: float, pre_max: float = 0.03,
                         post_max: float = 0.03, pre_avg: float = 0.1,
                         post_avg: float = 0.07, combine: float = 0.03,
                         delay: float = 0., frame_rate: int = 100):
    """
    Instantiate the peak picking for ODF post-processing.

    :param threshold: float
    :param pre_max: float, default = 0.03
    :param post_max: float, default = 0.03
    :param pre_avg: float, default = 0.1
    :param post_avg: float, default = 0.07
    :param combine: float, default = 0.03
    :param delay: float, default = 0.
    :param frame_rate: int, default = 100
    :return: peak_proc: OnsetPeakPickingProcessor
    """
    peak_proc = OnsetPeakPickingProcessor(
        threshold=threshold, pre_max=pre_max, post_max=post_max,
        pre_avg=pre_avg, post_avg=post_avg, combine=combine, delay=delay,
        fps=frame_rate)
    return peak_proc


def create_filter_reduction_processor(hrow: np.ndarray, hcol: np.ndarray,
                                      positive_diffs: bool = True):
    """
    Instantiate a reduction processor based on separable FIR filters for
    spectrogram reduction.

    :param hrow: np.ndarray
    :param hcol: np.ndarray
    :param positive_diffs: bool, default=True
    :return: filter_proc : SepFirProcessor
    """
    filter_proc = SeparableFilterSpectrogramDifference(
        hrow=hrow, hcol=hcol, positive_diffs=positive_diffs)
    return filter_proc


def preprocessing(audiofile: str, preprocessor: SequentialProcessor,
                  sample_rate: int = 44100):
    """
    Preprocess an audio file: Load it, convert it to mono, extract features.

    :param audiofile: str
    :param preprocessor: SequentialProcessor
    :param sample_rate: int, default=44100
    :return: log_filtered_spec: np.array, shape=(..., ...)
    """
    # Load and normalize audio file
    signal, sr = librosa.load(audiofile, sr=sample_rate, mono=True)

    # Calculate preprocessed spectrogram
    log_filtered_spec = np.array(preprocessor.process(signal))
    return log_filtered_spec


def reduction(log_filtered_spec: np.array, reduction_processor):
    """
    Reduce a spectrogram to a 1D onset detection function.

    :param log_filtered_spec: np.array, shape=(..., ...)
    :param reduction_processor: Processor from madmom
    :return: odf : np.array, shape=(..., )
    """
    # Calculate positive spectrogram differences
    diff_spec = reduction_processor.process(log_filtered_spec)
    # Calculate ODF by summing up spectral differences
    odf = diff_spec.sum(axis=1)
    return odf


def peak_picking(odf: np.array, peak_processor):
    """
    Find peaks in an onset detection function.

    :param odf: np.array, shape=(..., )
    :param peak_processor: Processor
    :return: onsets: list
    """
    # Peak-Picking
    onsets = peak_processor.process(odf)
    return onsets


def evaluation(onsets: list, annotations: list,
               evaluation_window: float = 0.05, combine: float = 0.03,
               delay: float = 0.0):
    """
    Evaluate onset detection performance on a single audio file.

    :param onsets: list
    :param annotations: list
    :param evaluation_window: float, default=0.05
    :param combine: float, default=0.03
    :param delay: float, default=0.0
    :return: evl: OnsetEvaluation
    """
    evl = OnsetEvaluation(detections=onsets, annotations=annotations,
                          window=evaluation_window, combine=combine,
                          delay=delay)
    return evl


def generate_start_position(time_filter_length: int = 3,
                            freq_filter_length: int = 3,
                            num_particles: int = 700, rand_min: float = -10,
                            rand_max: float = 10, random_state=42):
    """

    :param time_filter_length: int, default=3
    :param freq_filter_length: int, default=3
    :param num_particles: int, default=700
    :param rand_min: float, default=-10
    :param rand_max: float, default=10
    :param random_state: int, default=42
    :return: init_pos: np.array
    """

    rng = np.random.default_rng(random_state)
    particles = []
    # This could be improved by creating two random matrices for the filter
    # and for the threshold, respectively.
    for i in range(num_particles):
        arr = rng.uniform(
            rand_min, rand_max, freq_filter_length+time_filter_length)
        particle = np.append(arr, rng.uniform(0, 20)).tolist()
        particles.append(particle)
    init_pos = np.array(particles, dtype=np.float32)
    return init_pos


def cost_function(f: list, spec_list: list, annotation_list: list,
                  frame_rate: int = 200, pre_max: float = 0.03,
                  post_max: float = 0.03, pre_avg: float = 0.1,
                  post_avg: float = 0.07, evaluation_window: float = 0.05,
                  combine: float = 0.05, delay: float = 0,
                  freq_filter_length: int = 3, num_cores: int = -1):
    """

    :param f: list
    :param spec_list: list
    :param annotation_list: list
    :param frame_rate: int, default=200
    :param pre_max: float, default=0.03
    :param post_max: float, default=0.03
    :param pre_avg: float, default=0.1
    :param post_avg: float, default=0.07
    :param evaluation_window: float, default=0.05
    :param combine: float, default=0.05
    :param delay: float, default=0
    :param freq_filter_length: int, default=3
    :param num_cores: int, default=-1
    :return: c: float
    """
    c = Parallel(
        n_jobs=num_cores, verbose=10)(
        delayed(iterate_over_all_files)(
            spec_list=spec_list, annotation_list=annotation_list,
            f=f[particle_count], freq_filter_length=freq_filter_length,
            frame_rate=frame_rate, pre_max=pre_max, post_max=post_max,
            pre_avg=pre_avg, post_avg=post_avg,
            evaluation_window=evaluation_window, combine=combine, delay=delay)
        for particle_count in range(f.shape[0]))
    return c


def iterate_over_all_files(spec_list: list, annotation_list: list, f: list,
                           freq_filter_length: int = 3, frame_rate: int = 200,
                           pre_max: float = 0.03, post_max: float = 0.03,
                           pre_avg: float = 0.1, post_avg: float = 0.07,
                           evaluation_window: float = 0.05,
                           combine: float = 0.05, delay: float = 0):
    """

    :param spec_list: list
    :param annotation_list: list
    :param f: list
    :param freq_filter_length: int, default=3
    :param frame_rate: int, default=200
    :param pre_max: float, default=0.03
    :param post_max: float, default=0.03
    :param pre_avg: float, default=0.1
    :param post_avg: float, default=0.07
    :param evaluation_window: float, default=0.05
    :param combine: float, default=0.05
    :param delay: float, default=0
    :return: c: int
    """
    # Create Filter processor
    filter_proc = create_filter_reduction_processor(
        hrow=f[:freq_filter_length], hcol=f[freq_filter_length:-1],
        positive_diffs=True)

    # Create Peak-Picking-Processor
    peak_proc = create_peakprocessor(
        threshold=f[-1], pre_max=pre_max, post_max=post_max, pre_avg=pre_avg,
        post_avg=post_avg, combine=combine, delay=delay, frame_rate=frame_rate)

    # List containing all evaluation objects
    list_evals = [None] * len(spec_list)

    for file_count in range(len(spec_list)):
        odf = reduction(spec_list[file_count], reduction_processor=filter_proc)
        # Onset Detection with best threshold
        onsets = peak_picking(odf=odf, peak_processor=peak_proc)
        eval_obj = evaluation(
            onsets=onsets, annotations=annotation_list[file_count],
            evaluation_window=evaluation_window, combine=combine, delay=delay)
        list_evals[file_count] = eval_obj
    # Evaluate complete dataset
    score = OnsetSumEvaluation(list_evals).fmeasure
    c = 1 - score
    return c


def optimize_folds(folds: list, training_folds: list, test_fold: list,
                   preprocessor, path_to_folds: str = "data/splits",
                   audiodir: str = "data/audio", sample_rate: int = 44100,
                   time_filter_length: int = 3, freq_filter_length: int = 3,
                   num_particles: int = 700,
                   refdir: str = "data/annotations/onsets",
                   frame_rate: int = 200, pre_max: float = 0.03,
                   post_max: float = 0.03, pre_avg: float = 0.1,
                   post_avg: float = 0.07, evaluation_window: float = 0.05,
                   combine: float = 0.05, delay: float = 0,
                   num_iterations: int = 60, num_cores: int = -1):
    """

    :param folds: list
    :param training_folds: list
    :param test_fold: list
    :param path_to_folds: str, default="data/splits"
    :param audiodir: str, default="data/audio"
    :param sample_rate: int, default=44100
    :param preprocessor: Processor
    :param time_filter_length: int, default=3
    :param freq_filter_length: int, default=3
    :param num_particles: int, default=700
    :param refdir: str, default="data/annotations/onsets"
    :param frame_rate: int, default=200
    :param pre_max: float, default=0.03
    :param post_max: float, default=0.03
    :param pre_avg: float, default=0.1
    :param post_avg: float, default=0.07
    :param evaluation_window: float, default=0.05
    :param combine: float, default=0.05
    :param delay: float, default=0
    :param num_iterations: int, default=60
    :param num_cores: int, default=-1
    :return: f_score, pos
    """
    # Generate Training Set
    spec_list = []
    annotation_list = []
    count = 0
    track_names = []
    for fold in training_folds:
        count = count + 1
        with open(os.path.join(path_to_folds, fold)) as fd:
            # List containing all track names for a specific fold
            tracks = [line.rstrip() for line in fd]
            for track in tracks:
                track_names.append(track)
            fd.close()
        # Iterate over all tracks in this fold and perform preprocessing and
        # reduction steps
        for file in tqdm(tracks, desc=f"Preprocessing tracks in fold {count} "
                                      f"of {len(folds)}"):
            audiofile = os.path.join(audiodir, file + ".flac")
            spec = preprocessing(audiofile=audiofile, sample_rate=sample_rate,
                                 preprocessor=preprocessor)
            spec_list.append(spec)
            reffile = os.path.join(refdir, file + ".onsets")
            annotation_list.append(madmom.io.load_events(reffile))

    # Run PSO over training set
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.3}  # Set-up hyperparameters
    # Define particle dimension
    dimensions = time_filter_length + freq_filter_length + 1
    init_pos = generate_start_position(time_filter_length=time_filter_length,
                                       freq_filter_length=freq_filter_length,
                                       rand_min=-10, rand_max=10,
                                       num_particles=num_particles)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles, dimensions=dimensions, options=options,
        ftol=0.001, ftol_iter=10, init_pos=init_pos)
    kwargs = {
        'spec_list': spec_list, 'annotation_list': annotation_list,
        'frame_rate': frame_rate, 'pre_max': pre_max, 'post_max': post_max,
        'pre_avg': pre_avg, 'post_avg': post_avg,
        'evaluation_window': evaluation_window, 'combine': combine,
        'delay': delay, 'freq_filter_length': freq_filter_length,
        'num_cores': num_cores}

    cost, pos = optimizer.optimize(cost_function, iters=num_iterations,
                                   **kwargs)  # Perform optimization

    # Test on Test set
    # Preprocess data of remaining fold
    spec_list = []
    annotation_list = []
    with open(os.path.join(path_to_folds, test_fold)) as fd:
        # List containing all track names for a specific fold
        tracks = [line.rstrip() for line in fd]
        fd.close()
    # Iterate over all tracks in this fold
    for file in tqdm(tracks, "Preprocessing tracks in fold 8 of 8 (Test)"):
        audiofile = os.path.join(audiodir, file + ".flac")
        spec = preprocessing(audiofile=audiofile, sample_rate=sample_rate,
                             preprocessor=preprocessor)
        spec_list.append(spec)
        reffile = os.path.join(refdir, file + ".onsets")
        annotation_list.append(madmom.io.load_events(reffile))
    # Reduction step
    f_score = -(iterate_over_all_files(
        spec_list=spec_list, annotation_list=annotation_list, f=pos,
        freq_filter_length=freq_filter_length, frame_rate=frame_rate,
        pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg,
        evaluation_window=evaluation_window, combine=combine, delay=delay) - 1)
    return f_score, pos


def main(audiodir: str = "data/audio", refdir: str = "data/annotations/onsets",
         sample_rate: int = 44100, normalize_input: bool = True,
         frame_size: int = 2048, frame_rate: int = 200, num_bands: int = 24,
         fmin: float = 27.5, fmax: float = 16000, norm_filters: bool = False,
         time_filter_length: int = 3, freq_filter_length: int = 3,
         pre_max: float = 0.03, post_max: float = 0.03, pre_avg: float = 0.1,
         post_avg: float = 0.07, combine: float = 0.03, delay: float = 0,
         evaluation_window: float = 0.05, num_particles: int = 700,
         num_iterations: int = 60, path_to_folds: str = "data/splits",
         num_cores: int = -1):
    """

    :param audiodir: str, default="data/audio"
    :param refdir: str, default="data/annotations/onsets"
    :param sample_rate: int, default=44100
    :param normalize_input: bool, default=True
    :param frame_size: int, default=2048
    :param frame_rate: int, default=200
    :param num_bands: int, default=24
    :param fmin: float, default=27.5
    :param fmax: float, default=16000
    :param norm_filters: bool, default=False
    :param time_filter_length: int, default=3
    :param freq_filter_length: int, default=3
    :param pre_max: float, default=0.03
    :param post_max: float, default=0.03
    :param pre_avg: float, default=0.1
    :param post_avg: float, default=0.07
    :param combine: float, default=0.03
    :param delay: float, default=0
    :param evaluation_window: float, default=0.05
    :param num_particles: int, default=700
    :param num_iterations: int, default=60
    :param path_to_folds: str, default="data/splits"
    :param num_cores: int, default=-1
    :return:
    """
    # Logging
    logging.basicConfig(
        filename='report.log', encoding='utf-8', level=logging.DEBUG)
    logging.info(msg=f"#######################################################"
                     f"########################################")
    logging.info(msg=f"Started new Global optimization run! {num_particles} "
                     f"Particles, {num_iterations} Iterations")
    logging.info(msg=f"Freq. filter length: {freq_filter_length} Time filter "
                     f"length: {time_filter_length}")
    logging.info(msg="Running on " + str(num_cores) + " Cores...")

    # Instantiate Preprocessor with parser parameters
    preprocessor = create_preprocessor(
        sample_rate=sample_rate, normalize_input=normalize_input,
        frame_size=frame_size, frame_rate=frame_rate, num_bands=num_bands,
        fmin=fmin, fmax=fmax, norm_filters=norm_filters)

    # Run 8-fold cross validation PSO algorithm
    all_f_scores = []
    all_params = []
    for i in range(8):
        folds = []
        for file in os.listdir(path_to_folds):
            folds.append(file)
            print("Filename: " + str(file))
        all_folds = folds.copy()
        del folds[i]
        f_score, params = optimize_folds(
            folds=folds, training_folds=folds, test_fold=all_folds[i],
            path_to_folds=path_to_folds, audiodir=audiodir,
            sample_rate=sample_rate, preprocessor=preprocessor,
            time_filter_length=time_filter_length,
            freq_filter_length=freq_filter_length, num_particles=num_particles,
            refdir=refdir, frame_rate=frame_rate, pre_max=pre_max,
            post_max=post_max, pre_avg=pre_avg, post_avg=post_avg,
            evaluation_window=evaluation_window, combine=combine, delay=delay,
            num_iterations=num_iterations, num_cores=num_cores)
        all_f_scores.append(f_score)
        all_params.append(params)
        logging.info("F-Score on Test set: " + str(f_score))
    dump(all_params, 'results/PSO/8_fold_params.joblib')
    dump(all_f_scores, 'results/PSO/8_fold_f_scores.joblib')
    logging.info("F-Scores on Test Set: " + str(all_f_scores))
    logging.info("Params: " + str(all_params))
    logging.info("Mean F-Score: " + str(np.mean(all_f_scores)))

    return 0


if __name__ == "__main__":
    # Parser
    p = argparse.ArgumentParser(
        description='PSO of separable filters for onset detection')

    # Path parameters
    p.add_argument('-audiodir', '--audiodir', type=str, default='data/audio',
                   help='Path to audio data')
    p.add_argument(
        '-refdir', '--refdir', type=str, default='data/annotations/onsets',
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
    p.add_argument('-norm_filters', '--norm_filters', type=bool,
                   default=False,
                   help='Normalize the filter of the filterbank to area 1')

    # Peak picking and evaluation parameters
    p.add_argument(
        '-time_filter_length', '--time_filter_length', type=int, default=3,
        help='Length of the filter in time direction (must be odd integer)')
    p.add_argument(
        '-freq_filter_length', '--freq_filter_length', type=int, default=3,
        help='Length of the filter in frequency dir (must be odd integer)')
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

    p.add_argument('-path_to_folds', '--path_to_folds', type=str,
                   default='data/splits', help='Path to folds')

    # PSO arguments
    p.add_argument('-num_particles', '--num_particles', type=int, default=700,
                   help='Number of particles')
    p.add_argument('-num_iterations', '--num_iterations', type=int, default=60,
                   help='Number of iterations')
    p.add_argument('-num_cores', '--num_cores', type=int, default=-1,
                   help='Number of cores used for parallel computing')

    args = vars(p.parse_args())
    main(**args)
