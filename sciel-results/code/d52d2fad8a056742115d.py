"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [1, n] (1-indexed). signs is a list/array of -1/+1 of length n.
    Returns the max absolute sum (an integer).
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1

    # For each step d >= 1, and each residue r in [0, d-1],
    # the APs with step d starting at position r+1 (1-indexed) correspond to
    # contiguous subsequences of signs[r::d] (0-indexed).
    # We need max |sum| over all contiguous subsequences of each such sequence.
    # That is the max subarray sum problem on the sequence and its negation.
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            # Compute prefix sums
            csum = np.cumsum(seq)
            # Max subarray sum = max(csum) - min(csum[:i] for i) 
            # We can do this in O(L) with Kadane, or use prefix approach.
            # For efficiency with numpy, let's use a vectorized approach:
            # max_{i<=j} csum[j] - csum[i-1] where csum[0]=0
            # Actually, let's just use the standard method:
            # min_prefix = min of csum[0..j-1], max_sub = max of csum[j] - min_prefix
            # But we also need min subarray sum (most negative), so also track max_prefix.
            
            # Let's use a simple loop for correctness, but it might be slow.
            # Alternative: use scipy or a clever numpy trick.
            # For now, let's use an O(L) Python loop per sequence, but that's too slow.
            
            # Better: use the fact that max subarray sum can be computed as:
            # max_{j} csum[j] - min_{i<j} csum[i]
            # We can compute this with a running minimum.
            
            # Let's use a vectorized approach with numpy:
            # min_prefix_before[j] = min(csum[0..j-1]) for j=1..L
            # This is a bit tricky to vectorize directly.
            
            # For now, let's use a simple O(L) approach in Python, but only for sequences
            # where it's worth it. Actually, let's just use the prefix sum approach with
            # a running min/max in a loop, but in numpy style.
            
            # Actually, the simplest correct approach for small L (which is most cases)
            # is to just compute all subarray sums. But that's O(L^2).
            
            # Let's use Kadane's algorithm in a vectorized-ish way:
            # For max subarray:
            # current_max = max(x, current_max + x) for each x
            # This is hard to vectorize without a loop.
            
            # Given the constraints, let's use a different approach:
            # Precompute for each d, the sequences, and use a simple O(L) Kadane.
            # Total work is sum_{d=1}^{n-1} sum_{r=0}^{d-1} O(len(signs[r::d]))
            # = sum_{d=1}^{n-1} O(n) = O(n^2) which is ~65536, very fast.
            
            # Kadane for max subarray sum:
            best = -10**9
            current = 0
            for x in seq:
                current = max(x, current + x)
                if current > best:
                    best = current
            if best > max_disc:
                max_disc = best
            
            # Kadane for min subarray sum (most negative):
            best_min = 10**9
            current_min = 0
            for x in seq:
                current_min = min(x, current_min + x)
                if current_min < best_min:
                    best_min = current_min
            if -best_min > max_disc:
                max_disc = -best_min

    return max_disc

def _greedy_construct():
    """
    Construct signs greedily, choosing each sign to minimize the current
    maximum AP sum.
    """
    n = _N
    signs = [0] * n
    # Start with position 0
    signs[0] = 1
    
    for i in range(1, n):
        # Try both -1 and +1, pick the one that gives lower max AP sum
        best_val = 10**9
        best_sign = 1
        for s in (1, -1):
            signs[i] = s
            val = _max_abs_ap_sum(signs)
            if val < best_val:
                best_val = val
                best_sign = s
        signs[i] = best_sign
    return signs

def _local_search(signs, time_budget=50.0):
    """
    Local search: flip individual signs if it reduces the max AP sum.
    """
    n = _N
    signs = list(signs)
    current_max = _max_abs_ap_sum(signs)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            # Try flipping sign at position i
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = _max_abs_ap_sum(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
    return signs, current_max

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with greedy construction
    signs = _greedy_construct()
    current_max = _max_abs_ap_sum(signs)
    
    # Local search to improve
    if time.time() < deadline - 5.0:
        signs, current_max = _local_search(signs, time_budget=min(30.0, deadline - time.time() - 5.0))
    
    # If we have time, try a few random restarts
    rng = random.Random(42)
    best_signs = signs
    best_max = current_max
    
    while time.time() < deadline - 2.0:
        # Random start
        rand_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        rand_max = _max_abs_ap_sum(rand_signs)
        
        # Local search from random start
        if time.time() < deadline - 5.0:
            ls_signs, ls_max = _local_search(rand_signs, time_budget=min(15.0, deadline - time.time() - 5.0))
            if ls_max < best_max:
                best_max = ls_max
                best_signs = ls_signs
    
    return best_signs
# EVOLVE-BLOCK-END
