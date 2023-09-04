"""
Separable filter processor required to reproduce the results in the paper
'Separable 2D filters for note onset detection in music signals'.
"""
import scipy
import numpy as np
from madmom.processors import Processor


class SeparableFilterSpectrogramDifference(Processor):
    """
    Separable Filter Spectrogram Difference Processor class.

    Parameters
    ----------
    hrow : np.ndarray, default = np.array([1/3, 1/3, 1/3], dtype=float)
        Apply a linear filter with this kernel to the spectrogram in the
        frequency direction.
    hcol : np.ndarray, default = np.array([-1., 0., 1.], dtype=float)
        Apply a linear filter with this kernel to the spectrogram in the time
        direction.
    positive_diffs : bool, optional
        Keep only the positive differences, i.e. set all diff values < 0 to 0.
    """

    def __init__(self, *,
                 hrow: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3],
                                             dtype=float),
                 hcol: np.ndarray = np.array([-1., 0., 1.], dtype=float),
                 positive_diffs: bool = True):
        self.hrow = hrow
        self.hcol = hcol
        self.positive_diffs = positive_diffs

    def process(self, data: np.ndarray, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        Parameters
        ----------
        data : np.ndarray
            Data to be processed.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        diff : np.ndarray instance
            Spectrogram difference.
        """
        # Filter setup
        img = data.astype(np.float32)
        diff = scipy.signal.sepfir2d(img, self.hrow, self.hcol)
        if self.positive_diffs:
            diff = np.maximum(diff, 0)

        return diff


class TwoDimensionalSeparableFilterSpectrogramDifference(
    SeparableFilterSpectrogramDifference):
    """
    Separable Filter Spectrogram Difference Processor class.

    Parameters
    ----------
    hrow : np.ndarray, default = np.array([1/3, 1/3, 1/3], dtype=float)
        Apply a linear filter with this kernel to the spectrogram in the
        frequency direction.
    hcol : np.ndarray, default = np.array([-1., 0., 1.], dtype=float)
        Apply a linear filter with this kernel to the spectrogram in the time
        direction.
    positive_diffs : bool, optional
        Keep only the positive differences, i.e. set all diff values < 0 to 0.
    """

    def __init__(self, *,
                 hrow: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3],
                                             dtype=float),
                 hcol: np.ndarray = np.array([-1., 0., 1.], dtype=float),
                 positive_diffs: bool = True):
        super().__init__(hrow=hrow, hcol=hcol, positive_diffs=positive_diffs)
        self._2d_filter_kernel = np.multiply(self.hrow.reshape(-1, 1),
                                             self.hcol.reshape(1, -1))

    def process(self, data: np.ndarray, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        diff : np.ndarray instance
            Spectrogram difference.
        """
        # Filter setup
        img = data.astype(np.float32)
        diff = scipy.signal.convolve2d(img, self._2d_filter_kernel,
                                       mode="same", boundary="symm")
        if self.positive_diffs:
            diff = np.maximum(diff, 0)

        return diff
