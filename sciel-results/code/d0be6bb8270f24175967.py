import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

# Precompute LCM of 1..22
_LCM_22 = 1
for i in range(1, 23):
    _LCM_22 = math.lcm(_LCM_22, i)

# Precompute scaled reciprocals: v[i] = LCM // (i+1) for i in 0..21
# So v[0] corresponds to 1/1, v[1] to 1/2, etc.
_SCALED_VALS = np.array([_LCM_22 // (i + 1) for i in range(22)], dtype=np.int64)

def _is_dissociated_numpy(S):
    """
    Checks if S is dissociated using NumPy for speed.
    S is a list of integers (1-based).
    """
    if not S:
        return True
    
    # Get the scaled values for the elements in S
    # S contains 1..22, so indices are x-1
    indices = [x - 1 for x in S]
    v = _SCALED_VALS[indices]
    
    # Generate all subset sums
    # Start with {0}
    sums = np.array([0], dtype=np.int64)
    
    for val in v:
        # new_sums = sums + val
        # sums = concatenate(sums, new_sums)
        sums = np.concatenate((sums, sums + val))
        
    # Check for duplicates
    # Sort and check adjacent
    sums.sort()
    # If any adjacent elements are equal, there is a duplicate
    # We can use np.diff and check for 0
    if np.any(np.diff(sums) == 0):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    universe = list(range(1, 23))
    
    # We want to maximize cardinality.
    # Try sizes 18, 17, 16...
    
    # Bias for exclusions: smaller numbers are more likely to cause collisions.
    # So we want to EXCLUDE them.
    # We will sample exclusions with a bias towards small numbers.
    # Weights: 1/x is a good proxy for "risk".
    weights = [1.0 / x for x in universe]
    
    # Helper to sample exclusions
    def sample_exclusions(k):
        # Sample k distinct items from universe with weights
        # Python's random.choices doesn't do distinct sampling directly with weights easily
        # We can use a simple rejection sampling or weighted permutation
        # Since universe is small (22), we can just shuffle with bias?
        # Or use numpy.random.choice with replace=False
        # But numpy choice with replace=False and weights is supported in newer versions?
        # Let's use a simple method: assign random priority based on weight
        # Or just pick uniformly and rely on the fact that we try many times.
        # Actually, uniform sampling of exclusions is fine if we try enough.
        # But biased is better.
        
        # Simple weighted sampling without replacement:
        # Create a list of (weight, item)
        # Use a priority queue?
        # For small N, just do:
        selected = []
        remaining = list(universe)
        rem_weights = list(weights)
        
        for _ in range(k):
            # Pick one with probability proportional to weight
            # Use cumulative sum
            total_w = sum(rem_weights)
            r = random.random() * total_w
            cum = 0
            for i, w in enumerate(rem_weights):
                cum += w
                if r <= cum:
                    selected.append(remaining[i])
                    remaining.pop(i)
                    rem_weights.pop(i)
                    break
                    
        return selected

    # Try size 18
    # Time budget: 15s
    t_start = time.time()
    attempts = 0
    while time.time() < t_start + 15.0 and time.time() < deadline:
        exclusions = sample_exclusions(4)
        S = [x for x in universe if x not in exclusions]
        
        if _is_dissociated_numpy(S):
            return S
        attempts += 1
        
    # Try size 17
    # Time budget: 15s
    t_start = time.time()
    while time.time() < t_start + 15.0 and time.time() < deadline:
        exclusions = sample_exclusions(5)
        S = [x for x in universe if x not in exclusions]
        
        if _is_dissociated_numpy(S):
            return S
        attempts += 1
        
    # Try size 16
    # Time budget: 15s
    t_start = time.time()
    while time.time() < t_start + 15.0 and time.time() < deadline:
        exclusions = sample_exclusions(6)
        S = [x for x in universe if x not in exclusions]
        
        if _is_dissociated_numpy(S):
            return S
        attempts += 1
        
    # Fallback: Return a known valid set of size 16 or smaller
    # The previous program found size 16.
    # Let's return a hardcoded size 16 set if we have one, or just return the last one found?
    # We didn't save the last one if it was invalid.
    # Let's do a quick greedy construction to get a valid set.
    
    current_set = []
    for x in universe:
        candidate = current_set + [x]
        if _is_dissociated_numpy(candidate):
            current_set = candidate
            
    if current_set:
        return current_set
    else:
        return [1]

# EVOLVE-BLOCK-END
