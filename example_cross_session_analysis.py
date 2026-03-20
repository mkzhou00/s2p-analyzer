"""
Example usage of cross_session_analyzer module.

This script demonstrates how to use the cross-session analysis utilities
to track and analyze neurons across multiple recording sessions.
"""

import numpy as np
import pandas as pd
from s2p_utils import (
    filter_neurons_by_session_count,
    apply_ID_shift,
    load_neuron_table,
    get_neurons_present_in_all_sessions,
    get_session_neuron_indices,
    align_neuron_data_across_sessions,
    compute_session_stability,
    get_session_overlap_matrix,
)


def example_1_filter_neurons():
    """Example 1: Filter neurons based on session presence."""
    print("=" * 60)
    print("Example 1: Filtering neurons by session count")
    print("=" * 60)
    
    # Create a sample tracking table
    tracking_table = pd.DataFrame({
        'neuron_ID': list(range(1, 11)),
        'session_0': [0, 1, 2, 3, -1, 5, 6, -1, 8, 9],
        'session_1': [10, 11, -1, 13, 14, 15, -1, 17, 18, 19],
        'session_2': [20, -1, 22, 23, 24, -1, 26, 27, -1, 29],
        'session_3': [30, 31, 32, -1, 34, 35, 36, 37, 38, -1]
    })
    
    print("\nOriginal tracking table:")
    print(tracking_table)
    
    # Filter neurons present in all sessions
    all_sessions = get_neurons_present_in_all_sessions(tracking_table)
    print(f"\nNeurons present in all 4 sessions: {len(all_sessions)}")
    print(all_sessions)
    
    # Filter neurons present in at least 3 sessions
    at_least_3 = filter_neurons_by_session_count(tracking_table, count=3, exact=False)
    print(f"\nNeurons present in at least 3 sessions: {len(at_least_3)}")
    print(at_least_3[['neuron_ID']].values.flatten())
    
    print()


def example_2_session_overlap():
    """Example 2: Compute session overlap matrix."""
    print("=" * 60)
    print("Example 2: Computing session overlap")
    print("=" * 60)
    
    tracking_table = pd.DataFrame({
        'neuron_ID': list(range(1, 11)),
        'session_0': [0, 1, 2, 3, -1, 5, 6, -1, 8, 9],
        'session_1': [10, 11, -1, 13, 14, 15, -1, 17, 18, 19],
        'session_2': [20, -1, 22, 23, 24, -1, 26, 27, -1, 29],
    })
    
    overlap_matrix = get_session_overlap_matrix(tracking_table)
    print("\nSession overlap matrix (number of shared neurons):")
    print(overlap_matrix)
    print()


def example_3_align_and_analyze():
    """Example 3: Align neuron data and compute stability."""
    print("=" * 60)
    print("Example 3: Aligning neuron data and computing stability")
    print("=" * 60)
    
    # Create tracking table
    tracking_table = pd.DataFrame({
        'neuron_ID': [1, 2, 3, 4, 5],
        'session_0': [0, 1, 2, -1, 4],
        'session_1': [5, 6, -1, 8, 9],
        'session_2': [10, -1, 12, 13, 14]
    })
    
    print("\nTracking table:")
    print(tracking_table)
    
    # Simulate neuron activity data for each session
    # In practice, this would be loaded from suite2p output files
    np.random.seed(42)
    
    # Session 0: 5 neurons, 20 timepoints
    session_0_data = np.random.randn(5, 20)
    
    # Session 1: 10 neurons, 20 timepoints
    session_1_data = np.random.randn(10, 20)
    
    # Session 2: 15 neurons, 20 timepoints
    session_2_data = np.random.randn(15, 20)
    
    print(f"\nSession data shapes:")
    print(f"  Session 0: {session_0_data.shape}")
    print(f"  Session 1: {session_1_data.shape}")
    print(f"  Session 2: {session_2_data.shape}")
    
    # Align data across sessions
    aligned_data, neuron_ids = align_neuron_data_across_sessions(
        tracking_table,
        [session_0_data, session_1_data, session_2_data]
    )
    
    print(f"\nAligned data shape: {aligned_data.shape}")
    print(f"  (n_neurons={len(neuron_ids)}, n_sessions=3, n_timepoints=20)")
    
    # Compute stability for neurons present in multiple sessions
    stability = compute_session_stability(aligned_data, method='correlation')
    
    print("\nNeuron stability scores (average correlation across sessions):")
    for neuron_id, stab in zip(neuron_ids, stability):
        if not np.isnan(stab):
            print(f"  Neuron {neuron_id}: {stab:.3f}")
        else:
            print(f"  Neuron {neuron_id}: N/A (insufficient data)")
    
    print()


def example_4_id_shift():
    """Example 4: Apply ID shifts for neuron tracking."""
    print("=" * 60)
    print("Example 4: Applying ID shifts")
    print("=" * 60)
    
    # This is useful when combining tracking tables from different animals
    # where neuron indices need to be offset to avoid collisions
    
    tracking_table = pd.DataFrame({
        'neuron_ID': [1, 2, 3],
        'session_0': ['0', '1', '2'],
        'session_1': ['10', '11', '-1'],
        'session_2': ['20', '-1', '22']
    })
    
    print("\nOriginal tracking table:")
    print(tracking_table)
    
    # Apply a shift to session_1 (e.g., when merging data from multiple planes)
    tracking_table_shifted = tracking_table.copy()
    tracking_table_shifted['session_1'] = apply_ID_shift(
        tracking_table_shifted, 'session_1', shift_value=100
    )
    
    print("\nAfter applying +100 shift to session_1:")
    print(tracking_table_shifted)
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Cross-Session Neuron Analysis - Usage Examples")
    print("=" * 60 + "\n")
    
    example_1_filter_neurons()
    example_2_session_overlap()
    example_3_align_and_analyze()
    example_4_id_shift()
    
    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
