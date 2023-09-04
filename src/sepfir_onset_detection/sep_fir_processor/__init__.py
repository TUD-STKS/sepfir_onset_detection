"""
Separable filter processor required to reproduce the results in the paper
'Separable 2D filters for note onset detection in music signals'.
"""

from ._sep_fir_processor import (
    SeparableFilterSpectrogramDifference,
    TwoDimensionalSeparableFilterSpectrogramDifference)

__all__ = ["SeparableFilterSpectrogramDifference",
           "TwoDimensionalSeparableFilterSpectrogramDifference"]
