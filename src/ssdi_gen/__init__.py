"""SSDI-based federated data partition generator."""

from .core import compute_ssdi_metrics, theoretical_vmax
from .generate import (
    get_combo_params,
    generate_9_methods_and_analyse,
    generate_9_methods_and_analyse_structured,
    generate_ssdi_matrix,
    generate_ssdi_matrix_structured,
    inspect_structured_generation_plan,
)
from .plotting import generate_plots, generate_statistics, plot_single_matrix_distribution

__all__ = [
    "theoretical_vmax",
    "compute_ssdi_metrics",
    "get_combo_params",
    "generate_ssdi_matrix",
    "generate_9_methods_and_analyse",
    "generate_ssdi_matrix_structured",
    "generate_9_methods_and_analyse_structured",
    "generate_statistics",
    "generate_plots",
    "plot_single_matrix_distribution",
    "inspect_structured_generation_plan",
]
