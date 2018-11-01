"""skmlm implements MLM models."""

from .lssvr import LSSVR, RegENN_LSSVR, RegCNN_LSSVR, DiscENN_LSSVR, MI_LSSVR, AM_LSSVR

__all__ = ['LSSVR', 'RegENN_LSSVR', 'RegCNN_LSSVR', 'DiscENN_LSSVR', 'MI_LSSVR', 'AM_LSSVR']

__version__ = '0.0.1'