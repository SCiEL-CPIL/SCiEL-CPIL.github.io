"""Low discrepancy on all arithmetic progressions - optimized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n]."""
    n = len(signs)
    signs_arr = np.array(signs, dtype=np.int8)
    max_abs = 0
    
    for d in range(1, n + 1):
        max_len = (n - 1) // d + 1
        
        # Build AP matrix: rows are residue classes
        ap_matrix = np.zeros((d, max_len), dtype=np.int8)
        lengths = np.empty(d, dtype=np.int32)
        for r in range(d):
            seq = signs_arr[r::d]
            lengths[r] = len(seq)
            ap_matrix[r, :len(seq)] = seq
        
        # Prefix sums
        P = np.zeros((d, max_len + 1), dtype=np.int32)
        if max_len > 0:
            P[:, 1:] = np.cumsum(ap_matrix, axis=1)
        
        # Mask invalid parts
        indices = np.arange(max_len + 1)
        valid_mask = indices[None, :] <= lengths[:, None]
        
        # Max subarray sum
        P_for_min = np.where(valid_mask, P, np.inf)
        running_min = np.minimum.accumulate(P_for_min, axis=1)
        max_diff = P - running_min
        max_diff_masked = np.where(valid_mask, max_diff, -np.inf)
        max_subarray_val = np.max(max_diff_masked)
        
        # Min subarray sum
        P_for_max = np.where(valid_mask, P, -np.inf)
        running_max = np.maximum.accumulate(P_for_max, axis=1)
        min_diff = P - running_max
        min_diff_masked = np.where(valid_mask, min_diff, np.inf)
        min_subarray_val = np.min(min_diff_masked)
        
        current_max_abs = max(max_subarray_val, -min_subarray_val)
        if current_max_abs > max_abs:
            max_abs = current_max_abs
            # Early termination if we already exceed a known bound
            if max_abs > 20:
                return int(max_abs)
    
    return int(max_abs)

def _thue_morse(n):
    """Generate Thue-Morse sequence as ±1."""
    signs = []
    for i in range(n):
        # Count bits
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    return signs

def _random_signs(n, seed):
    rng = random.Random(seed)
    return [rng.choice((-1, 1)) for _ in range(n)]

def _block_flip(signs, i, length, rng):
    """Flip a contiguous block of signs."""
    new_signs = signs[:]
    for j in range(i, min(i + length, len(signs))):
        new_signs[j] = -signs[j]
    return new_signs

def construct():
    deadline = time.time() + 85  # 5s safety margin
    n = 256
    
    # Multiple diverse initializations
    best_signs = None
    best_val = float('inf')
    
    # Thue-Morse
    tm = _thue_morse(n)
    v = _max_ap_discrepancy(tm)
    if v < best_val:
        best_val = v
        best_signs = tm[:]
    
    # Random initializations
    for seed in range(20):
        signs = _random_signs(n, 1000 + seed)
        v = _max_ap_discrepancy(signs)
        if v < best_val:
            best_val = v
            best_signs = signs[:]
    
    # Local search with simulated annealing on best initialization
    signs = best_signs[:]
    val = best_val
    
    # First pass: greedy single-flip local search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            old = signs[i]
            signs[i] = -old
            v = _max_ap_discrepancy(signs)
            if v < val:
                val = v
                improved = True
                if val < best_val:
                    best_val = val
                    best_signs = signs[:]
            else:
                signs[i] = old
    
    # Second pass: simulated annealing with single flips
    if time.time() < deadline - 5:
        temp = 2.0
        cooling = 0.999
        signs = best_signs[:]
        val = best_val
        rng = random.Random(123)
        
        while time.time() < deadline - 2:
            i = rng.randrange(n)
            old = signs[i]
            signs[i] = -old
            v = _max_ap_discrepancy(signs)
            delta = v - val
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                val = v
                if val < best_val:
                    best_val = val
                    best_signs = signs[:]
            else:
                signs[i] = old
            temp *= cooling
    
    # Third pass: block flips
    if time.time() < deadline - 2:
        signs = best_signs[:]
        val = best_val
        rng = random.Random(456)
        temp = 1.5
        cooling = 0.9995
        
        while time.time() < deadline - 1:
            # Random block flip
            i = rng.randrange(n)
            length = rng.choice([1, 2, 3, 4, 5])
            old_block = signs[i:i+length]
            for j in range(i, min(i + length, n)):
                signs[j] = -signs[j]
            v = _max_ap_discrepancy(signs)
            delta = v - val
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                val = v
                if val < best_val:
                    best_val = val
                    best_signs = signs[:]
            else:
                for j in range(i, min(i + length, n)):
                    signs[j] = old_block[j - i]
            temp *= cooling
    
    # Final greedy pass
    if time.time() < deadline - 0.5:
        signs = best_signs[:]
        val = best_val
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                old = signs[i]
                signs[i] = -old
                v = _max_ap_discrepancy(signs)
                if v < val:
                    val = v
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_signs = signs[:]
                else:
                    signs[i] = old
    
    return best_signs
# EVOLVE-BLOCK-END
