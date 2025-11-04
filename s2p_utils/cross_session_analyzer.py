"""
Cross-session neuron analysis utilities.

This module provides utilities for tracking and analyzing neurons across
multiple recording sessions. It includes functions for:
- Filtering neurons based on presence across sessions
- Applying ID shifts for neuron tracking
- Loading and aligning neuron data across sessions
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Sequence, Iterable, Any, List, Dict, Tuple
import re


def filter_neurons_by_session_count(
    df: pd.DataFrame,
    count: int,
    session_cols: Optional[Sequence] = None,
    absent_values: Optional[Iterable[Any]] = None,
    exact: bool = True
) -> pd.DataFrame:
    """
    Filter neurons (rows) by the exact or at least number of sessions in which they appear.

    Assumptions:
    - The first column is the neuron_ID.
    - Remaining columns are session columns (0..n).
    - A neuron is absent in a session if the cell is NaN, -1 (int), or "-1" (str).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing neuron IDs and session information.
    count : int
        Exact number of sessions the neuron must appear in (or at least if exact is False).
    session_cols : Optional[Sequence], optional
        List of session column names; defaults to all except the first column.
    absent_values : Optional[Iterable[Any]], optional
        Iterable of values that indicate absence; defaults to [-1, "-1"].
    exact : bool, optional
        If True, filter for exact count; if False, filter for at least count. Default is True.

    Returns
    -------
    pd.DataFrame
        A filtered copy of df containing only rows with exactly or at least `count` present sessions.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'neuron_ID': [1, 2, 3, 4],
    ...     'session_0': [0, 1, 2, 3],
    ...     'session_1': [10, 11, -1, 13],
    ...     'session_2': [20, -1, -1, 23]
    ... })
    >>> filter_neurons_by_session_count(df, count=3, exact=True)
       neuron_ID  session_0  session_1  session_2
    0          1          0         10         20
    3          4          3         13         23
    >>> filter_neurons_by_session_count(df, count=2, exact=False)
       neuron_ID  session_0  session_1  session_2
    0          1          0         10         20
    1          2          1         11         -1
    3          4          3         13         23
    """
    if session_cols is None:
        session_cols = df.columns[1:]  # exclude neuron_ID column

    if absent_values is None:
        absent_values = (-1, "-1")

    sess = df[session_cols]

    # Build mask of "absent" entries: NaN OR equals any of the absent_values
    absent_mask = sess.isna()
    for v in absent_values:
        absent_mask = absent_mask | sess.eq(v)

    # Present if not absent
    present_counts = (~absent_mask).sum(axis=1)

    if exact:
        sel = present_counts.eq(count)
    else:
        sel = present_counts.ge(count)  # Use greater than or equal for at least

    return df.loc[sel].copy()


def apply_ID_shift(
    df: pd.DataFrame,
    col: str,
    shift_value: int,
    skip_value: Any = -1
) -> pd.Series:
    """
    Add shift_value to all numeric string values in a column.
    
    Skips skip_value (can be int or str) and any strings containing letters.
    This is useful for aligning neuron IDs across sessions when indices need to be offset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to update.
    col : str
        Column name to apply the shift to.
    shift_value : int
        Amount to add to numeric values.
    skip_value : Any, optional
        Value to skip (default -1). Can be int or str.

    Returns
    -------
    pd.Series
        A new Series with shifted values.

    Examples
    --------
    >>> df = pd.DataFrame({'session_1': ['0', '1', '2', '-1', 'invalid']})
    >>> apply_ID_shift(df, 'session_1', shift_value=100)
    0        100
    1        101
    2        102
    3         -1
    4    invalid
    Name: session_1, dtype: object
    """
    def transform(x):
        # Handle missing values (NaN, None etc.)
        if pd.isna(x):
            return x
        # Compare x to skip_value as both int and str
        if x == skip_value or str(x) == str(skip_value):
            return x
        x_str = str(x)
        # Only shift if the string is a valid integer (including negative)
        if re.fullmatch(r"-?\d+", x_str):
            return str(int(x_str) + shift_value)
        return x

    new_col = df[col].apply(transform)
    return new_col


def load_neuron_table(
    file_path: str,
    apply_shifts: Optional[Dict[str, int]] = None,
    skip_value: Any = -1
) -> pd.DataFrame:
    """
    Load a neuron tracking table from CSV and optionally apply ID shifts.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the neuron tracking table.
    apply_shifts : Optional[Dict[str, int]], optional
        Dictionary mapping column names to shift values to apply.
        Example: {'session_1': 100, 'session_2': 200}
    skip_value : Any, optional
        Value to skip when applying shifts (default -1).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with shifts applied if specified.

    Examples
    --------
    >>> table = load_neuron_table('ROI_table.csv', apply_shifts={'session_1': 100})
    """
    df = pd.read_csv(file_path)
    
    if apply_shifts:
        for col, shift in apply_shifts.items():
            if col in df.columns:
                df[col] = apply_ID_shift(df, col, shift, skip_value)
    
    return df


def get_neurons_present_in_all_sessions(
    df: pd.DataFrame,
    session_cols: Optional[Sequence] = None,
    absent_values: Optional[Iterable[Any]] = None
) -> pd.DataFrame:
    """
    Filter neurons that are present in all sessions.
    
    This is a convenience wrapper around filter_neurons_by_session_count
    that automatically determines the total number of sessions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing neuron IDs and session information.
    session_cols : Optional[Sequence], optional
        List of session column names; defaults to all except the first column.
    absent_values : Optional[Iterable[Any]], optional
        Iterable of values that indicate absence; defaults to [-1, "-1"].

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only neurons present in all sessions.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'neuron_ID': [1, 2, 3],
    ...     'session_0': [0, 1, 2],
    ...     'session_1': [10, -1, 12],
    ...     'session_2': [20, -1, 22]
    ... })
    >>> get_neurons_present_in_all_sessions(df)
       neuron_ID  session_0  session_1  session_2
    0          1          0         10         20
    2          3          2         12         22
    """
    if session_cols is None:
        session_cols = df.columns[1:]
    
    num_sessions = len(session_cols)
    return filter_neurons_by_session_count(
        df, count=num_sessions, session_cols=session_cols,
        absent_values=absent_values, exact=True
    )


def get_session_neuron_indices(
    df: pd.DataFrame,
    session_col: str,
    absent_values: Optional[Iterable[Any]] = None
) -> np.ndarray:
    """
    Get the neuron indices for a specific session from the tracking table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing neuron tracking information.
    session_col : str
        Name of the session column.
    absent_values : Optional[Iterable[Any]], optional
        Values that indicate neuron absence; defaults to [-1, "-1", NaN].

    Returns
    -------
    np.ndarray
        Array of neuron indices present in the session.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'neuron_ID': [1, 2, 3],
    ...     'session_0': ['0', '1', '-1']
    ... })
    >>> get_session_neuron_indices(df, 'session_0')
    array([0, 1])
    """
    if absent_values is None:
        absent_values = (-1, "-1")
    
    col_data = df[session_col]
    
    # Build mask for absent values
    absent_mask = col_data.isna()
    for v in absent_values:
        absent_mask = absent_mask | col_data.eq(v)
    
    # Get indices where neurons are present
    present_mask = ~absent_mask
    indices = col_data[present_mask].astype(int).values
    
    return indices


def align_neuron_data_across_sessions(
    tracking_table: pd.DataFrame,
    session_data_list: List[np.ndarray],
    session_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Align neuron activity data across sessions using a tracking table.

    Parameters
    ----------
    tracking_table : pd.DataFrame
        DataFrame with neuron_ID in first column and session indices in remaining columns.
    session_data_list : List[np.ndarray]
        List of neuron activity arrays for each session. Each array should have
        shape (n_neurons, ...) where n_neurons can differ across sessions.
    session_cols : Optional[List[str]], optional
        List of session column names. Defaults to all columns except the first.

    Returns
    -------
    aligned_data : np.ndarray
        Aligned data array with shape (n_tracked_neurons, n_sessions, ...).
        Contains NaN for neurons not present in a session.
    neuron_ids : List[int]
        List of neuron IDs corresponding to rows in aligned_data.

    Examples
    --------
    >>> tracking_table = pd.DataFrame({
    ...     'neuron_ID': [1, 2],
    ...     'session_0': [0, 1],
    ...     'session_1': [5, -1]
    ... })
    >>> session_0_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 neurons, 2 timepoints
    >>> session_1_data = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], 
    ...                             [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])  # 6 neurons
    >>> aligned, ids = align_neuron_data_across_sessions(
    ...     tracking_table, [session_0_data, session_1_data]
    ... )
    >>> aligned.shape
    (2, 2, 2)
    >>> ids
    [1, 2]
    """
    if session_cols is None:
        session_cols = list(tracking_table.columns[1:])
    
    n_sessions = len(session_cols)
    n_tracked_neurons = len(tracking_table)
    
    if n_sessions != len(session_data_list):
        raise ValueError(
            f"Number of session columns ({n_sessions}) must match "
            f"number of session data arrays ({len(session_data_list)})"
        )
    
    # Determine the data shape from the first session
    example_shape = session_data_list[0].shape[1:]
    
    # Initialize aligned data array with NaN
    aligned_data = np.full(
        (n_tracked_neurons, n_sessions, *example_shape),
        np.nan
    )
    
    neuron_ids = tracking_table.iloc[:, 0].tolist()
    
    # Fill in data for each session
    for session_idx, (session_col, session_data) in enumerate(
        zip(session_cols, session_data_list)
    ):
        for neuron_idx, row in tracking_table.iterrows():
            cell_idx = row[session_col]
            
            # Check if neuron is present in this session
            if pd.notna(cell_idx) and cell_idx != -1 and str(cell_idx) != "-1":
                cell_idx = int(cell_idx)
                if cell_idx < len(session_data):
                    aligned_data[neuron_idx, session_idx] = session_data[cell_idx]
    
    return aligned_data, neuron_ids


def compute_session_stability(
    aligned_data: np.ndarray,
    method: str = 'correlation'
) -> np.ndarray:
    """
    Compute stability metrics for neurons across sessions.

    Parameters
    ----------
    aligned_data : np.ndarray
        Aligned neuron data with shape (n_neurons, n_sessions, n_features).
    method : str, optional
        Method to compute stability. Options: 'correlation', 'variance'.
        Default is 'correlation'.

    Returns
    -------
    np.ndarray
        Stability metric for each neuron. Shape (n_neurons,).

    Examples
    --------
    >>> data = np.array([[[1, 2], [1.1, 2.1]], [[3, 4], [3.1, 4.1]]])
    >>> stability = compute_session_stability(data, method='correlation')
    >>> stability.shape
    (2,)
    """
    n_neurons, n_sessions, n_features = aligned_data.shape
    stability = np.zeros(n_neurons)
    
    for neuron_idx in range(n_neurons):
        neuron_data = aligned_data[neuron_idx]  # shape: (n_sessions, n_features)
        
        # Skip neurons with missing data
        valid_sessions = ~np.isnan(neuron_data).any(axis=1)
        if valid_sessions.sum() < 2:
            stability[neuron_idx] = np.nan
            continue
        
        valid_data = neuron_data[valid_sessions]
        
        if method == 'correlation':
            # Compute average pairwise correlation between sessions
            correlations = []
            for i in range(len(valid_data)):
                for j in range(i + 1, len(valid_data)):
                    corr = np.corrcoef(valid_data[i], valid_data[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            stability[neuron_idx] = np.mean(correlations) if correlations else np.nan
            
        elif method == 'variance':
            # Compute coefficient of variation across sessions
            session_means = np.mean(valid_data, axis=1)
            if np.mean(session_means) != 0:
                cv = np.std(session_means) / np.mean(session_means)
                stability[neuron_idx] = 1 - min(cv, 1)  # Convert to stability (higher is better)
            else:
                stability[neuron_idx] = np.nan
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return stability


def get_session_overlap_matrix(
    df: pd.DataFrame,
    session_cols: Optional[Sequence] = None,
    absent_values: Optional[Iterable[Any]] = None
) -> pd.DataFrame:
    """
    Compute a matrix showing the number of shared neurons between each pair of sessions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing neuron tracking information.
    session_cols : Optional[Sequence], optional
        List of session column names; defaults to all except the first column.
    absent_values : Optional[Iterable[Any]], optional
        Values indicating neuron absence; defaults to [-1, "-1", NaN].

    Returns
    -------
    pd.DataFrame
        Symmetric matrix with session names as index and columns,
        values represent number of shared neurons.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'neuron_ID': [1, 2, 3],
    ...     'session_0': [0, 1, -1],
    ...     'session_1': [10, 11, 12],
    ...     'session_2': [20, -1, 22]
    ... })
    >>> get_session_overlap_matrix(df)
              session_0  session_1  session_2
    session_0          2          2          1
    session_1          2          3          2
    session_2          1          2          2
    """
    if session_cols is None:
        session_cols = list(df.columns[1:])
    
    if absent_values is None:
        absent_values = (-1, "-1")
    
    n_sessions = len(session_cols)
    overlap_matrix = np.zeros((n_sessions, n_sessions), dtype=int)
    
    # Build presence masks for each session
    presence_masks = []
    for session_col in session_cols:
        col_data = df[session_col]
        absent_mask = col_data.isna()
        for v in absent_values:
            absent_mask = absent_mask | col_data.eq(v)
        presence_masks.append(~absent_mask)
    
    # Compute overlap for each pair
    for i in range(n_sessions):
        for j in range(n_sessions):
            if i == j:
                # Diagonal: count neurons present in this session
                overlap_matrix[i, j] = presence_masks[i].sum()
            else:
                # Off-diagonal: count neurons present in both sessions
                overlap_matrix[i, j] = (presence_masks[i] & presence_masks[j]).sum()
    
    return pd.DataFrame(
        overlap_matrix,
        index=session_cols,
        columns=session_cols
    )
