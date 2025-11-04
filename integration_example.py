"""
Integration example: Cross-session analysis with main_analyzer.py

This script demonstrates how to integrate cross-session neuron tracking
with the existing Suite2p analysis pipeline in main_analyzer.py.
"""

import os
import numpy as np
import pandas as pd
from s2p_utils.data_loader import DataLoader
from s2p_utils.cross_session_analyzer import (
    filter_neurons_by_session_count,
    align_neuron_data_across_sessions,
    compute_session_stability,
    get_session_overlap_matrix,
)


def analyze_neurons_across_sessions(
    main_folder: str,
    animal: str,
    days: list,
    num_planes_list: list,
    num_flyback_list: list,
    imaging_system: str,
    tracking_table_path: str,
    min_sessions: int = 2
):
    """
    Analyze neurons tracked across multiple recording sessions.
    
    Parameters
    ----------
    main_folder : str
        Path to main data folder containing all animals
    animal : str
        Animal identifier
    days : list
        List of day numbers to analyze
    num_planes_list : list
        Number of planes for each day
    num_flyback_list : list
        Number of flyback frames for each day
    imaging_system : str
        Imaging system type ('Bruker' or 'INSS')
    tracking_table_path : str
        Path to neuron tracking table CSV (e.g., from ROICAT)
    min_sessions : int
        Minimum number of sessions a neuron must be present in
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print("=" * 60)
    print(f"Cross-Session Analysis for {animal}")
    print("=" * 60)
    
    # Load tracking table
    print(f"\n1. Loading tracking table from: {tracking_table_path}")
    tracking_table = pd.read_csv(tracking_table_path)
    print(f"   Total tracked neurons: {len(tracking_table)}")
    
    # Filter neurons present in minimum number of sessions
    print(f"\n2. Filtering neurons present in at least {min_sessions} sessions...")
    filtered_table = filter_neurons_by_session_count(
        tracking_table, count=min_sessions, exact=False
    )
    print(f"   Neurons meeting criteria: {len(filtered_table)}")
    
    # Compute session overlap
    print("\n3. Computing session overlap matrix...")
    overlap_matrix = get_session_overlap_matrix(filtered_table)
    print("   Session overlap (number of shared neurons):")
    print(overlap_matrix)
    
    # Load processed data for each day
    print("\n4. Loading neuron activity data for each session...")
    session_data = []
    
    for id, day in enumerate(days):
        data_dir = os.path.join(main_folder, animal, f"d{day}")
        print(f"   Loading day {day}...")
        
        # Check if processed data exists
        F_file = os.path.join(data_dir, "files", "F_5hz.npy")
        if os.path.exists(F_file):
            # Load pre-processed 5Hz data
            Fcorr_5hz = np.load(F_file, allow_pickle=True)
            print(f"     Loaded 5Hz data: {len(Fcorr_5hz)} planes")
            
            # Concatenate planes if multiple
            if isinstance(Fcorr_5hz, np.ndarray) and len(Fcorr_5hz) > 1:
                # Stack all planes together
                Fcorr_combined = np.vstack([Fcorr_5hz[ip] for ip in range(len(Fcorr_5hz))])
            else:
                Fcorr_combined = Fcorr_5hz[0] if len(Fcorr_5hz) > 0 else Fcorr_5hz
            
            session_data.append(Fcorr_combined)
            print(f"     Shape: {Fcorr_combined.shape}")
        else:
            print(f"     Warning: Processed data not found for day {day}")
            print(f"     Expected: {F_file}")
    
    if len(session_data) != len(days):
        print(f"\n   Error: Could not load data for all sessions")
        return None
    
    # Align data across sessions
    print("\n5. Aligning neuron data across sessions...")
    aligned_data, neuron_ids = align_neuron_data_across_sessions(
        filtered_table,
        session_data
    )
    print(f"   Aligned data shape: {aligned_data.shape}")
    print(f"   (n_neurons={len(neuron_ids)}, n_sessions={len(days)}, n_timepoints={aligned_data.shape[2]})")
    
    # Compute stability metrics
    print("\n6. Computing neuron stability across sessions...")
    stability = compute_session_stability(aligned_data, method='correlation')
    
    # Filter out neurons with insufficient data
    valid_stability = stability[~np.isnan(stability)]
    print(f"   Neurons with valid stability: {len(valid_stability)}")
    
    if len(valid_stability) > 0:
        print(f"   Mean stability: {np.mean(valid_stability):.3f}")
        print(f"   Std stability: {np.std(valid_stability):.3f}")
        print(f"   Min stability: {np.min(valid_stability):.3f}")
        print(f"   Max stability: {np.max(valid_stability):.3f}")
        
        # Identify stable neurons (top 20%)
        stability_threshold = np.percentile(valid_stability, 80)
        stable_mask = stability >= stability_threshold
        n_stable = stable_mask.sum()
        print(f"\n   Highly stable neurons (top 20%): {n_stable}")
    
    # Save results
    results = {
        'tracking_table': filtered_table,
        'overlap_matrix': overlap_matrix,
        'aligned_data': aligned_data,
        'neuron_ids': neuron_ids,
        'stability': stability,
    }
    
    # Save to file
    result_dir = os.path.join(main_folder, animal, 'cross_session_results')
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n7. Saving results to: {result_dir}")
    np.save(os.path.join(result_dir, 'aligned_data.npy'), aligned_data)
    np.save(os.path.join(result_dir, 'stability.npy'), stability)
    filtered_table.to_csv(os.path.join(result_dir, 'filtered_neurons.csv'), index=False)
    overlap_matrix.to_csv(os.path.join(result_dir, 'session_overlap.csv'))
    
    print("\n" + "=" * 60)
    print("Cross-session analysis complete!")
    print("=" * 60)
    
    return results


def main():
    """
    Example usage of cross-session analysis integration.
    
    Note: This is a template. Update the paths and parameters
    to match your actual data structure.
    """
    
    # Example parameters (update these for your data)
    main_folder = "Z:\\2p\\experiment1"  # Update this path
    animal = "MZ_CA1_WD_F3"
    days = [1, 2, 3, 4]  # Days to analyze
    num_planes_list = np.ones(len(days), dtype=int) * 4
    num_flyback_list = np.ones(len(days), dtype=int) * 0
    imaging_system = "Bruker"
    
    # Path to tracking table (generated by ROICAT or similar)
    tracking_table_path = os.path.join(
        main_folder, animal, "ROI_table_full_plane0.csv"
    )
    
    # Check if tracking table exists
    if not os.path.exists(tracking_table_path):
        print(f"Error: Tracking table not found at: {tracking_table_path}")
        print("\nTo generate a tracking table:")
        print("1. Use ROICAT to track neurons across sessions")
        print("2. Save the output as ROI_table_full_plane0.csv")
        print("3. Update the tracking_table_path in this script")
        return
    
    # Run cross-session analysis
    results = analyze_neurons_across_sessions(
        main_folder=main_folder,
        animal=animal,
        days=days,
        num_planes_list=num_planes_list,
        num_flyback_list=num_flyback_list,
        imaging_system=imaging_system,
        tracking_table_path=tracking_table_path,
        min_sessions=2  # Require neurons to be present in at least 2 sessions
    )
    
    if results:
        print("\nAnalysis completed successfully!")
        print("\nResults saved:")
        print("  - aligned_data.npy: Aligned neuron activity across sessions")
        print("  - stability.npy: Stability scores for each neuron")
        print("  - filtered_neurons.csv: Filtered tracking table")
        print("  - session_overlap.csv: Session overlap matrix")


if __name__ == "__main__":
    main()
