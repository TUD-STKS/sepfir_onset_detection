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
from madmom.evaluation.onsets import OnsetEvaluation
from madmom.processors import SequentialProcessor
from sepfir_onset_detection.sep_fir_processor import \
    SeparableFilterSpectrogramDifference
import matplotlib.pyplot as plt
import seaborn as sns


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
threshold = 0.4  # Threshold for peak-picking
pre_max = 0.03  # Use pre_max seconds past information for moving maximum
post_max = 0.03  # Use post_max seconds future information for moving maximum
pre_avg = 0.1  # Use pre_avg seconds past information for moving average
post_avg = 0.07  # Use post_avg seconds future information for moving average
eval_window = 0.05  # Time window around a reference onset
combine = 0.03  # Only report one onset within combine seconds
delay = 0  # Report the detected onsets delay seconds delayed

# Instantiate Processors with above parameters
signal_proc = SignalProcessor(sample_rate=sr, num_channels=1, norm=True)
frames_proc = FramedSignalProcessor(frame_size=fs, fps=fps)
stft_proc = ShortTimeFourierTransformProcessor(window=window)
fil_spec_proc = FilteredSpectrogramProcessor(num_bands=num_bands, fmin=f_min,
                                             fmax=f_max, norm_filters=False)
logfil_spec_proc = LogarithmicSpectrogramProcessor()
specDiffProc = SpectrogramDifferenceProcessor(diff_max_bins=diff_max_bins,
                                              positive_diffs=True)
peak_proc = OnsetPeakPickingProcessor(threshold=threshold,
                                      pre_max=pre_max, post_max=post_max,
                                      pre_avg=pre_avg, post_avg=post_avg,
                                      combine=combine, delay=delay, fps=fps)

preprocessor = SequentialProcessor([signal_proc, frames_proc, stft_proc,
                                    fil_spec_proc, logfil_spec_proc])
filter_proc = SeparableFilterSpectrogramDifference(
    hrow=np.array([1/3, 1/3, 1/3], dtype=np.float32),
    hcol=np.array([1.0, 0, -1.0], dtype=np.float32), positive_diffs=False)

audiofile = (
    "data/audio/ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1.flac")
annotationsfile = ("data/annotations/onsets/ah_development_guitar_2684_"
                   "TexasMusicForge_Dandelion_pt1.onsets")


def preprocessing(audiofile, sr, preprocessor):
    # Load and normalize audio file
    signal, sr = librosa.load(audiofile, sr=sr, mono=True)

    # Calculate preprocessed spectrogram
    log_filtered_spec = preprocessor.process(signal)
    return log_filtered_spec


def reduction_filtering(log_filtered_spec):
    deriv = filter_proc.process(log_filtered_spec)
    odf = deriv.sum(axis=1)
    # Plot
    fig, axs = plt.subplots()
    sns.heatmap(data=deriv.T, ax=axs, cbar_kws={'label': 'Amplitude'})
    axs.invert_yaxis()
    axs.set_xlabel("Frame index")
    axs.set_ylabel("Logarithmic Filterbank Output")
    axs.set_title("")
    plt.tight_layout()
    return odf


def reduction_spectral_flux(log_filtered_spec):
    return madmom.features.onsets.spectral_flux(log_filtered_spec,
                                                diff_frames=None)


def peak_picking(odf, peakprocessor):
    # Peak-Picking
    onsets = peakprocessor.process(odf)
    return onsets


def evaluation(onsets, annotationsfile):
    # Evaluation
    # Load annotations
    annotations = madmom.io.load_events(annotationsfile)
    evl = OnsetEvaluation(detections=onsets, annotations=annotations,
                          window=eval_window, combine=combine, delay=delay)
    return evl


spec = preprocessing(audiofile=audiofile, sr=sr, preprocessor=preprocessor)
odf = reduction_filtering(log_filtered_spec=spec)
# onsets = peak_picking(odf=odf, peakprocessor=peak_proc)
# eval_obj = evaluation(onsets=onsets, annotationsfile=annotationsfile)
# print(eval_obj)
#
# fig, axs = plt.subplots()
# sns.lineplot(data=odf, ax=axs)
# axs.set_xlabel("Frame index")
# axs.set_ylabel("ODF")
# axs.set_title("ODF")
# plt.tight_layout()
plt.show()
