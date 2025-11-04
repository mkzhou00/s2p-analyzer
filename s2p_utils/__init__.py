"""
Suite2p utilities package for neuron data analysis.
"""

from .cross_session_analyzer import (
    filter_neurons_by_session_count,
    apply_ID_shift,
    load_neuron_table,
    get_neurons_present_in_all_sessions,
    get_session_neuron_indices,
    align_neuron_data_across_sessions,
    compute_session_stability,
    get_session_overlap_matrix,
)

__all__ = [
    'filter_neurons_by_session_count',
    'apply_ID_shift',
    'load_neuron_table',
    'get_neurons_present_in_all_sessions',
    'get_session_neuron_indices',
    'align_neuron_data_across_sessions',
    'compute_session_stability',
    'get_session_overlap_matrix',
]
