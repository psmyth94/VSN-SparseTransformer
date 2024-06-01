# ruff: noqa
from .clr import CLRAdam, CLRSchedule
from .layers import (
    GatedLinearUnit,
    LinearUnit,
    VariableSelection,
    SparseAttention,
    GatedResidualNetwork,
)
from .utils import entmax
