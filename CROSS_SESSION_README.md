# Cross-Session Neuron Analysis

This module provides utilities for tracking and analyzing neurons across multiple recording sessions in Suite2p data.

## Overview

The `cross_session_analyzer` module enables researchers to:
- Track neurons across multiple imaging sessions
- Filter neurons based on presence criteria
- Align neuron activity data across sessions
- Compute stability metrics for tracked neurons
- Analyze session overlap and shared neurons

## Installation

The module requires the following dependencies:
```bash
pip install numpy pandas scipy scikit-learn
```

## Quick Start

```python
from s2p_utils import (
    filter_neurons_by_session_count,
    align_neuron_data_across_sessions,
    compute_session_stability,
)

# Load your neuron tracking table (e.g., from ROICAT output)
import pandas as pd
tracking_table = pd.read_csv('ROI_table.csv')

# Filter neurons present in at least 3 sessions
stable_neurons = filter_neurons_by_session_count(
    tracking_table, count=3, exact=False
)

# Align neuron data across sessions
aligned_data, neuron_ids = align_neuron_data_across_sessions(
    tracking_table, 
    [session_0_data, session_1_data, session_2_data]
)

# Compute stability metrics
stability = compute_session_stability(aligned_data, method='correlation')
```

## Key Functions

### 1. Filter Neurons by Session Count

```python
filter_neurons_by_session_count(df, count, session_cols=None, 
                                 absent_values=None, exact=True)
```

Filter neurons based on the number of sessions in which they appear.

**Parameters:**
- `df`: DataFrame with neuron tracking information
- `count`: Number of sessions required
- `exact`: If True, filter for exactly `count` sessions; if False, at least `count`

**Example:**
```python
# Get neurons present in all sessions
all_sessions = get_neurons_present_in_all_sessions(tracking_table)

# Get neurons present in at least 2 sessions
at_least_2 = filter_neurons_by_session_count(
    tracking_table, count=2, exact=False
)
```

### 2. Apply ID Shifts

```python
apply_ID_shift(df, col, shift_value, skip_value=-1)
```

Add an offset to neuron indices in a specific session. Useful when combining data from multiple planes or animals.

**Example:**
```python
# Shift session_1 indices by 100
tracking_table['session_1'] = apply_ID_shift(
    tracking_table, 'session_1', shift_value=100
)
```

### 3. Align Neuron Data Across Sessions

```python
align_neuron_data_across_sessions(tracking_table, session_data_list, 
                                   session_cols=None)
```

Align neuron activity data across multiple sessions using a tracking table.

**Parameters:**
- `tracking_table`: DataFrame with neuron IDs and session indices
- `session_data_list`: List of numpy arrays containing neuron activity data
- `session_cols`: Optional list of session column names

**Returns:**
- `aligned_data`: Array with shape (n_neurons, n_sessions, ...)
- `neuron_ids`: List of neuron IDs

**Example:**
```python
# Load session data from Suite2p outputs
session_0_F = np.load('session_0/F.npy')
session_1_F = np.load('session_1/F.npy')
session_2_F = np.load('session_2/F.npy')

# Align across sessions
aligned_data, neuron_ids = align_neuron_data_across_sessions(
    tracking_table,
    [session_0_F, session_1_F, session_2_F]
)
```

### 4. Compute Session Stability

```python
compute_session_stability(aligned_data, method='correlation')
```

Compute stability metrics for neurons across sessions.

**Parameters:**
- `aligned_data`: Aligned neuron data with shape (n_neurons, n_sessions, n_features)
- `method`: Stability metric to compute ('correlation' or 'variance')

**Returns:**
- Array of stability scores for each neuron

**Example:**
```python
# Compute correlation-based stability
stability = compute_session_stability(aligned_data, method='correlation')

# Find most stable neurons
stable_neuron_indices = np.argsort(stability)[-10:]  # Top 10
```

### 5. Get Session Overlap Matrix

```python
get_session_overlap_matrix(df, session_cols=None, absent_values=None)
```

Compute a matrix showing the number of shared neurons between each pair of sessions.

**Example:**
```python
overlap_matrix = get_session_overlap_matrix(tracking_table)
print("Neurons shared between sessions:")
print(overlap_matrix)
```

## Complete Workflow Example

Here's a complete example workflow for analyzing neurons across sessions:

```python
import numpy as np
import pandas as pd
from s2p_utils import (
    load_neuron_table,
    get_neurons_present_in_all_sessions,
    align_neuron_data_across_sessions,
    compute_session_stability,
    get_session_overlap_matrix,
)

# 1. Load tracking table (from ROICAT or similar)
tracking_table = pd.read_csv('path/to/ROI_table.csv')

# 2. Check session overlap
overlap = get_session_overlap_matrix(tracking_table)
print(f"Session overlap:\n{overlap}")

# 3. Filter for neurons present in all sessions
stable_neurons = get_neurons_present_in_all_sessions(tracking_table)
print(f"Found {len(stable_neurons)} neurons in all sessions")

# 4. Load neuron activity data for each session
session_data = []
for session in ['session_0', 'session_1', 'session_2']:
    F = np.load(f'{session}/plane0/F.npy')
    # Apply neuropil correction and normalization as needed
    session_data.append(F)

# 5. Align data across sessions
aligned_data, neuron_ids = align_neuron_data_across_sessions(
    stable_neurons,
    session_data
)

# 6. Compute stability metrics
stability = compute_session_stability(aligned_data, method='correlation')

# 7. Identify stable neurons
stable_threshold = 0.5
stable_mask = stability > stable_threshold
print(f"Neurons with stability > {stable_threshold}: {stable_mask.sum()}")

# 8. Analyze stable neurons
stable_neuron_data = aligned_data[stable_mask]
# Perform further analysis on stable neurons...
```

## Integration with Existing Code

The module integrates seamlessly with existing Suite2p analysis pipelines:

```python
# In your main_analyzer.py or similar script
from s2p_utils.data_loader import DataLoader
from s2p_utils.cross_session_analyzer import (
    align_neuron_data_across_sessions,
    compute_session_stability,
)

# Load data for each day
days = [1, 2, 3, 4]
session_data = []

for day in days:
    data_dir = os.path.join(main_folder, animal, f"d{day}")
    data_loader = DataLoader(data_dir, num_planes, num_flyback, imaging_system)
    
    # Load and preprocess F
    F_corr = np.load(os.path.join(data_dir, "files", "F_5hz.npy"))
    session_data.append(F_corr)

# Load tracking table and analyze
tracking_table = pd.read_csv(os.path.join(main_folder, animal, "ROI_table.csv"))
aligned_data, neuron_ids = align_neuron_data_across_sessions(
    tracking_table, session_data
)
```

## Data Format Requirements

### Tracking Table Format

The tracking table should be a CSV file with the following structure:

```
neuron_ID,session_0,session_1,session_2,...
1,0,10,20
2,1,11,-1
3,2,-1,22
...
```

- First column: Unique neuron ID across all sessions
- Remaining columns: Neuron index in each session
- Missing neurons indicated by `-1` or `NaN`

### Neuron Activity Data Format

Each session's neuron activity data should be:
- NumPy array with shape `(n_neurons, n_timepoints)` or `(n_neurons, n_features)`
- Can be raw fluorescence (F), neuropil-corrected (F_corr), or normalized data
- Number of neurons can differ across sessions

## Testing

Run the test suite to verify functionality:

```bash
python test_cross_session_analyzer.py
```

Or run the examples:

```bash
python example_cross_session_analysis.py
```

## Use Cases

### 1. Longitudinal Studies

Track how individual neurons respond to stimuli across multiple days or weeks:

```python
# Analyze stimulus responses over time
stable_neurons = get_neurons_present_in_all_sessions(tracking_table)
aligned_responses = align_neuron_data_across_sessions(
    stable_neurons, daily_responses
)
# Compare responses across days
```

### 2. Learning and Plasticity

Study changes in neural representations during learning:

```python
# Compare neural activity before and after learning
pre_learning_neurons = filter_neurons_by_session_count(
    tracking_table, count=2, exact=True
)
# Analyze changes in response properties
```

### 3. Quality Control

Identify and exclude unstable neurons:

```python
stability = compute_session_stability(aligned_data)
stable_mask = stability > 0.5
high_quality_neurons = tracking_table[stable_mask]
```

## Tips and Best Practices

1. **Neuron Tracking**: Use ROICAT or similar tools to generate the tracking table
2. **Data Preprocessing**: Apply consistent preprocessing (neuropil correction, normalization) across sessions
3. **Quality Control**: Filter neurons based on stability metrics before analysis
4. **Memory Efficiency**: For large datasets, process data in batches
5. **Missing Data**: Functions handle missing neurons gracefully (indicated as NaN)

## Troubleshooting

**Issue**: "Number of session columns must match number of session data arrays"
- **Solution**: Ensure the tracking table has the correct number of session columns

**Issue**: "IndexError: index out of bounds"
- **Solution**: Check that neuron indices in the tracking table match the actual data dimensions

**Issue**: All stability values are NaN
- **Solution**: Ensure neurons are present in at least 2 sessions and data is not all zeros

## References

- Suite2p: https://github.com/MouseLand/suite2p
- ROICAT: https://github.com/RichieHakim/ROICaT

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New functions include docstrings with examples
- Tests are added for new functionality
- Documentation is updated

## License

This code is part of the s2p-analyzer project.
