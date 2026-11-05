<<<<<<< SEARCH
"""Size-efficient sorting networks on 17 wires – search-based construction."""

import time
import random
import numpy as np

N = 17

# Deadline guard: kill search well before 90s wall-clock limit
_START_TIME = time.time()
_DEADLINE = 75.0  # seconds; leaves 15s safety margin


def _now():
    return time.time()


def _time_exceeded():
    return (_now() - _START_TIME) >= _DEADLINE


def _verify_booleans_numpy(network):
    """
    Vectorized verification using NumPy.
    Checks all 2^N Boolean inputs.
    """
    total = 1 << N
    # Create indices 0 to total-1
    indices = np.arange(total, dtype=np.uint64)
    
    # Initialize wires: shape (total, N)
    # wire[k] for input m is bit k of m
    wires = np.zeros((total, N), dtype=np.uint8)
    for k in range(N):
        wires[:, k] = (indices >> k) & 1
        
    # Apply comparators
    for (i, j) in network:
        a = wires[:, i]
        b = wires[:, j]
        # min and max
        min_ab = np.minimum(a, b)
        max_ab = np.maximum(a, b)
        wires[:, i] = min_ab
        wires[:, j] = max_ab
        
    # Check sorted
    # For each row, check if wire[k] <= wire[k+1] for all k
    # This is equivalent to checking if any wire[k] > wire[k+1]
    # diff = np.diff(wires, axis=1)
    # return not np.any(diff < 0)
    
    # More efficient: check if all adjacent pairs are sorted
    # wires[:, :-1] <= wires[:, 1:]
    return np.all(wires[:, :-1] <= wires[:, 1:])
=======
"""Size-efficient sorting networks on 17 wires – search-based construction."""

import time
import random
import numpy as np

N = 17

# Deadline guard: kill search well before 90s wall-clock limit
_START_TIME = time.time()
_DEADLINE = 75.0  # seconds; leaves 15s safety margin


def _now():
    return time.time()


def _time_exceeded():
    return (_now() - _START_TIME) >= _DEADLINE


# Precompute all 2^N boolean inputs as bit vectors
_TOTAL_INPUTS = 1 << N
_INPUTS = np.arange(_TOTAL_INPUTS, dtype=np.uint64)
_WIRES_INIT = np.zeros((_TOTAL_INPUTS, N), dtype=np.uint8)
for _k in range(N):
    _WIRES_INIT[:, _k] = (_INPUTS >> _k) & 1


def _verify_booleans_fast(network):
    """
    Fast vectorized verification using precomputed inputs.
    Checks all 2^N Boolean inputs.
    """
    wires = _WIRES_INIT.copy()
    
    for (i, j) in network:
        a = wires[:, i]
        b = wires[:, j]
        # min and max using bitwise operations (faster than np.minimum/maximum for uint8)
        wires[:, i] = a & b
        wires[:, j] = a | b
        
    # Check sorted: no adjacent pair has wire[k] > wire[k+1]
    # For binary values, this means no pattern 10 in adjacent wires
    # Equivalent to: wires[:, :-1] <= wires[:, 1:]
    return np.all(wires[:, :-1] <= wires[:, 1:])
>>>>>>> REPLACE

