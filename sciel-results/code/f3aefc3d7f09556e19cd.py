"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_cardinality_separation(subset_sums_by_k):
    """Check that no sum appears in two different cardinality classes."""
    seen = {}
    for k, sums in subset_sums_by_k.items():
        for s in sums:
            if s in seen and seen[s] != k:
                return False
            seen[s] = k
    return True

def _compute_subset_sums_by_cardinality(arr):
    """Compute all subset sums grouped by cardinality using DP with bitsets."""
    n = len(arr)
    max_sum = sum(arr)
    # bitsets[k] is a Python int where bit i is 1 if sum i is achievable with cardinality k
    bitsets = [0] * (n + 1)
    bitsets[0] = 1  # sum 0 achievable with cardinality 0
    for a in arr:
        # Update in reverse to avoid using a element twice
        for k in range(n, 0, -1):
            bitsets[k] |= (bitsets[k-1] << a)
    # Convert bitsets to sets of sums
    result = {}
    for k in range(n + 1):
        sums = set()
        bs = bitsets[k]
        while bs:
            # Extract lowest set bit
            lsb = bs & (-bs)
            idx = lsb.bit_length() - 1
            sums.add(idx)
            bs ^= lsb
        result[k] = sums
    return result

def _is_valid(arr):
    """Check if the array satisfies all constraints."""
    if len(arr) == 0:
        return True
    # Distinct integers in 1..256
    if len(set(arr)) != len(arr):
        return False
    if any(x < 1 or x > 256 for x in arr):
        return False
    # Cardinality-separated subset sums
    subset_sums = _compute_subset_sums_by_cardinality(arr)
    return _check_cardinality_separation(subset_sums)

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    best_result = None
    best_size = 0
    
    # Strategy: Try geometric-like sequences with varying parameters
    # a_i = floor(alpha * r^i) for i = 0, 1, ..., n-1
    # We want to maximize n subject to all values in [1, 256] and validity
    
    # First, try pure powers of 2 and variations
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128 -> 8 elements
    # Can we do better?
    
    # Let's try sequences where a_i grows slightly slower than 2x
    # The key constraint is cardinality separation
    
    # Try various geometric progressions
    best_candidate = None
    
    # Approach 1: Greedy construction
    # Start with small values, add next smallest value that maintains validity
    def greedy_construct():
        arr = []
        for candidate in range(1, 257):
            if candidate in arr:
                continue
            test_arr = arr + [candidate]
            if _is_valid(test_arr):
                arr = test_arr
            if time.time() > deadline:
                break
        return arr
    
    # Approach 2: Structured geometric sequences
    def try_geometric(alpha, r, max_n=60):
        arr = []
        for i in range(max_n):
            val = int(round(alpha * (r ** i)))
            if val < 1 or val > 256:
                break
            if val in arr:
                continue
            arr.append(val)
        return arr
    
    # Try many geometric parameters
    for r in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]:
        for alpha in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16]:
            if time.time() > deadline:
                return best_result if best_result else list(range(1, 9))
            arr = try_geometric(alpha, r)
            if len(arr) > best_size and _is_valid(arr):
                best_size = len(arr)
                best_result = arr
    
    # Also try greedy
    if time.time() < deadline:
        greedy_arr = greedy_construct()
        if len(greedy_arr) > best_size:
            best_size = len(greedy_arr)
            best_result = greedy_arr
    
    # Ensure we return something valid
    if best_result is None or not _is_valid(best_result):
        # Fallback: powers of 2
        best_result = [2**i for i in range(8)]  # 1,2,4,8,16,32,64,128
    
    # Convert to plain Python ints and list
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
