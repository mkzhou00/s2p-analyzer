"""
Test script for cross_session_analyzer module.

This script tests the basic functionality of the cross-session analysis utilities.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import s2p_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s2p_utils.cross_session_analyzer import (
    filter_neurons_by_session_count,
    apply_ID_shift,
    get_neurons_present_in_all_sessions,
    get_session_neuron_indices,
    align_neuron_data_across_sessions,
    compute_session_stability,
    get_session_overlap_matrix,
)


def test_filter_neurons_by_session_count():
    """Test filtering neurons by session count."""
    print("Testing filter_neurons_by_session_count...")
    
    # Create test data
    df = pd.DataFrame({
        'neuron_ID': [1, 2, 3, 4, 5],
        'session_0': [0, 1, 2, 3, 4],
        'session_1': [10, 11, -1, 13, 14],
        'session_2': [20, -1, -1, 23, 24]
    })
    
    # Test exact count = 3
    result = filter_neurons_by_session_count(df, count=3, exact=True)
    assert len(result) == 3, f"Expected 3 neurons with 3 sessions, got {len(result)}"
    assert set(result['neuron_ID']) == {1, 4, 5}, "Wrong neurons filtered"
    print("  ✓ Exact count filtering works")
    
    # Test at least count = 2
    result = filter_neurons_by_session_count(df, count=2, exact=False)
    assert len(result) == 4, f"Expected 4 neurons with at least 2 sessions, got {len(result)}"
    assert set(result['neuron_ID']) == {1, 2, 4, 5}, "Wrong neurons filtered"
    print("  ✓ Minimum count filtering works")
    
    print("✓ filter_neurons_by_session_count tests passed\n")


def test_apply_ID_shift():
    """Test applying ID shifts to neuron indices."""
    print("Testing apply_ID_shift...")
    
    df = pd.DataFrame({
        'session_1': ['0', '1', '2', '-1', 'invalid', np.nan]
    })
    
    result = apply_ID_shift(df, 'session_1', shift_value=100, skip_value=-1)
    
    assert result.iloc[0] == '100', f"Expected '100', got {result.iloc[0]}"
    assert result.iloc[1] == '101', f"Expected '101', got {result.iloc[1]}"
    assert result.iloc[2] == '102', f"Expected '102', got {result.iloc[2]}"
    assert result.iloc[3] == '-1', f"Expected '-1', got {result.iloc[3]}"
    assert result.iloc[4] == 'invalid', f"Expected 'invalid', got {result.iloc[4]}"
    assert pd.isna(result.iloc[5]), f"Expected NaN, got {result.iloc[5]}"
    
    print("  ✓ ID shift applied correctly")
    print("✓ apply_ID_shift tests passed\n")


def test_get_neurons_present_in_all_sessions():
    """Test filtering neurons present in all sessions."""
    print("Testing get_neurons_present_in_all_sessions...")
    
    df = pd.DataFrame({
        'neuron_ID': [1, 2, 3, 4],
        'session_0': [0, 1, 2, 3],
        'session_1': [10, -1, 12, 13],
        'session_2': [20, -1, 22, 23]
    })
    
    result = get_neurons_present_in_all_sessions(df)
    assert len(result) == 3, f"Expected 3 neurons in all sessions, got {len(result)}"
    assert set(result['neuron_ID']) == {1, 3, 4}, "Wrong neurons filtered"
    
    print("  ✓ Neurons present in all sessions identified correctly")
    print("✓ get_neurons_present_in_all_sessions tests passed\n")


def test_get_session_neuron_indices():
    """Test getting neuron indices for a specific session."""
    print("Testing get_session_neuron_indices...")
    
    df = pd.DataFrame({
        'neuron_ID': [1, 2, 3, 4],
        'session_0': ['0', '5', '10', '-1']
    })
    
    indices = get_session_neuron_indices(df, 'session_0')
    expected = np.array([0, 5, 10])
    
    assert np.array_equal(indices, expected), f"Expected {expected}, got {indices}"
    
    print("  ✓ Session neuron indices extracted correctly")
    print("✓ get_session_neuron_indices tests passed\n")


def test_align_neuron_data_across_sessions():
    """Test aligning neuron data across sessions."""
    print("Testing align_neuron_data_across_sessions...")
    
    # Create tracking table
    tracking_table = pd.DataFrame({
        'neuron_ID': [1, 2, 3],
        'session_0': [0, 1, -1],
        'session_1': [5, -1, 7]
    })
    
    # Create session data
    session_0_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 neurons, 2 timepoints
    session_1_data = np.array([
        [10.0, 11.0], [12.0, 13.0], [14.0, 15.0],
        [16.0, 17.0], [18.0, 19.0], [20.0, 21.0],
        [22.0, 23.0], [24.0, 25.0]
    ])  # 8 neurons
    
    aligned, ids = align_neuron_data_across_sessions(
        tracking_table, [session_0_data, session_1_data]
    )
    
    assert aligned.shape == (3, 2, 2), f"Expected shape (3, 2, 2), got {aligned.shape}"
    assert ids == [1, 2, 3], f"Expected [1, 2, 3], got {ids}"
    
    # Check neuron 1 (present in both sessions)
    assert np.array_equal(aligned[0, 0], session_0_data[0]), "Neuron 1 session 0 mismatch"
    assert np.array_equal(aligned[0, 1], session_1_data[5]), "Neuron 1 session 1 mismatch"
    
    # Check neuron 2 (only in session 0)
    assert np.array_equal(aligned[1, 0], session_0_data[1]), "Neuron 2 session 0 mismatch"
    assert np.all(np.isnan(aligned[1, 1])), "Neuron 2 session 1 should be NaN"
    
    # Check neuron 3 (only in session 1)
    assert np.all(np.isnan(aligned[2, 0])), "Neuron 3 session 0 should be NaN"
    assert np.array_equal(aligned[2, 1], session_1_data[7]), "Neuron 3 session 1 mismatch"
    
    print("  ✓ Neuron data aligned correctly across sessions")
    print("✓ align_neuron_data_across_sessions tests passed\n")


def test_compute_session_stability():
    """Test computing session stability metrics."""
    print("Testing compute_session_stability...")
    
    # Create test data with different correlation patterns
    data = np.array([
        [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],  # Neuron 1: high stability (similar pattern)
        [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]],  # Neuron 2: lower stability (different pattern)
    ])
    
    stability = compute_session_stability(data, method='correlation')
    
    assert len(stability) == 2, f"Expected 2 stability values, got {len(stability)}"
    assert stability[0] > 0.9, f"Expected high stability for neuron 1, got {stability[0]}"
    assert stability[1] < stability[0], f"Neuron 2 should have lower stability than neuron 1 (got {stability[1]} vs {stability[0]})"
    
    print("  ✓ Session stability computed correctly")
    print("✓ compute_session_stability tests passed\n")


def test_get_session_overlap_matrix():
    """Test computing session overlap matrix."""
    print("Testing get_session_overlap_matrix...")
    
    df = pd.DataFrame({
        'neuron_ID': [1, 2, 3, 4],
        'session_0': [0, 1, -1, 3],
        'session_1': [10, 11, 12, -1],
        'session_2': [20, -1, 22, 23]
    })
    
    overlap_matrix = get_session_overlap_matrix(df)
    
    assert overlap_matrix.shape == (3, 3), f"Expected shape (3, 3), got {overlap_matrix.shape}"
    
    # Check diagonal (number of neurons in each session)
    assert overlap_matrix.loc['session_0', 'session_0'] == 3  # neurons 1, 2, 4
    assert overlap_matrix.loc['session_1', 'session_1'] == 3  # neurons 1, 2, 3
    assert overlap_matrix.loc['session_2', 'session_2'] == 3  # neurons 1, 3, 4
    
    # Check off-diagonal (shared neurons)
    assert overlap_matrix.loc['session_0', 'session_1'] == 2  # neurons 1 and 2
    assert overlap_matrix.loc['session_0', 'session_2'] == 2  # neurons 1 and 4
    assert overlap_matrix.loc['session_1', 'session_2'] == 2  # neurons 1 and 3
    
    print("  ✓ Session overlap matrix computed correctly")
    print("✓ get_session_overlap_matrix tests passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running cross_session_analyzer tests")
    print("=" * 60 + "\n")
    
    try:
        test_filter_neurons_by_session_count()
        test_apply_ID_shift()
        test_get_neurons_present_in_all_sessions()
        test_get_session_neuron_indices()
        test_align_neuron_data_across_sessions()
        test_compute_session_stability()
        test_get_session_overlap_matrix()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
