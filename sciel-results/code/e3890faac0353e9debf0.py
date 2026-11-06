<<<<<<< SEARCH
"""Size-efficient sorting networks on 17 wires via bit-sliced search."""
import time
import random
import sys

N = 17
ALL_BITS = (1 << N) - 1
DEADLINE = 80.0  # Safety margin before 90s kill
START_TIME = time.time()

# Precompute the initial state for all 2^N inputs.
# State is a list of N integers, where state[k] is a bitmask of length 2^N.
# Bit v of state[k] is 1 if the v-th input has a 1 on wire k.
# Input v is an integer 0..2^N-1.
# Wire k of input v is (v >> k) & 1.
# So for each wire k, we need a mask where bit v is set if (v >> k) & 1.

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

# Precompute initial wire masks
# mask[k] has bit v set if the k-th bit of v is 1.
INITIAL_MASKS = []
for k in range(N):
    mask = 0
    for v in range(1 << N):
        if (v >> k) & 1:
            mask |= (1 << v)
    INITIAL_MASKS.append(mask)

def verify_network(comps):
    """
    Verify network using bit-sliced simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    # State: list of N integers, each 2^N bits wide.
    # Copy initial masks
    state = INITIAL_MASKS[:]
    
    for i, j in comps:
        wi = state[i]
        wj = state[j]
        
        # Comparator (i, j): min to i, max to j.
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        new_wi = wi & wj
        new_wj = wi | wj
        
        state[i] = new_wi
        state[j] = new_wj
        
    # Check if sorted.
    # A binary vector is sorted if it is non-decreasing.
    # This means for all k, wire k <= wire k+1.
    # In bitmask terms: state[k] & ~state[k+1] must be 0.
    # i.e., state[k] is a subset of state[k+1].
    for k in range(N - 1):
        if state[k] & ~state[k+1]:
            return False
            
    return True
=======
"""Size-efficient sorting networks on 17 wires via NumPy vectorized search."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 55.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute initial states for all 2^N inputs using NumPy.
# Shape: (NUM_INPUTS, N), dtype: uint8
# input_states[input_idx, wire_idx] = 1 if wire has 1, else 0
def _init_states():
    # Generate all input vectors
    # We can use np.arange and bit operations, or just iterate.
    # np.arange is fast.
    inputs = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Expand to 2D: shape (NUM_INPUTS, N)
    # We want bit k of inputs to be in column k.
    # (inputs[:, None] >> np.arange(N)) & 1
    states = (inputs[:, None] >> np.arange(N)) & 1
    return states.astype(np.uint8)

INITIAL_STATES = _init_states()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using NumPy vectorized simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        # Check if initial states are sorted (only true for N=1 or specific cases, usually false for N>1)
        # For N=17, initial state is not sorted.
        return False
        
    # Copy initial states
    # We need a mutable copy. .copy() is fast enough.
    state = INITIAL_STATES.copy()
    
    for i, j in comps:
        wi = state[:, i]
        wj = state[:, j]
        
        # Comparator (i, j): min to i, max to j.
        # Vectorized min/max
        new_wi = np.minimum(wi, wj)
        new_wj = np.maximum(wi, wj)
        
        state[:, i] = new_wi
        state[:, j] = new_wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This is equivalent to checking if any state[:, k] > state[:, k+1]
    # diff = state[:, 1:] - state[:, :-1]
    # if np.any(diff < 0): return False
    # Faster: check if state[:, k] & ~state[:, k+1] has any 1s?
    # Actually, since values are 0 or 1:
    # sorted if state[:, k] <= state[:, k+1] for all k.
    # This is true if there is no index where state[:, k] == 1 and state[:, k+1] == 0.
    # i.e., ~np.any(state[:, :-1] & (1 - state[:, 1:]))
    
    # Let's use the subtraction method, it's very fast in NumPy
    diffs = state[:, 1:] - state[:, :-1]
    if np.any(diffs < 0):
        return False
        
    return True
>>>>>>> REPLACE

