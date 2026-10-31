import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150s kill

def _score_from_vector(vec):
    """
    Compute the score for a binary vector representing the set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses FFT for fast computation.
    """
    # Find max element
    nonzero_indices = np.nonzero(vec)[0]
    if len(nonzero_indices) < 2:
        return 0.0
    max_val = nonzero_indices[-1]
    if max_val < 1:
        return 0.0
    
    # Determine FFT length
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # Pad vector
    if len(vec) < fft_len:
        padded_vec = np.zeros(fft_len, dtype=np.float64)
        padded_vec[:len(vec)] = vec
    else:
        padded_vec = vec[:fft_len].astype(np.float64)
        
    f = np.fft.fft(padded_vec)
    
    # Sumset |U+U|
    conv = np.fft.ifft(f * f).real
    sums_count = np.sum(np.abs(conv) > 1e-9)
    
    # Difference set |U-U|
    # Circular correlation. Since we padded to 2L-1, the circular correlation
    # matches the linear correlation for the supported range.
    # The zero lag is at index 0.
    # Positive lags 1..max_val are at indices 1..max_val.
    # Negative lags -1..-max_val are at indices fft_len-1 .. fft_len-max_val.
    # All non-zero entries in the array correspond to valid differences.
    corr = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(corr) > 1e-9)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _vec_to_list(vec):
    return [int(i) for i in np.nonzero(vec)[0]]

def _list_to_vec(lst, max_val):
    vec = np.zeros(max_val + 1, dtype=np.float64)
    for u in lst:
        if u <= max_val:
            vec[u] = 1.0
    return vec

def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    # We will search for sets U of size n.
    # The score depends on the ratio R = |U-U|/|U+U| and M = max(U).
    # For random sets, R is close to 1. For Sidon sets, R ~ 2.
    # We want to find sets with R > 2 or M very small.
    # Let's try various sizes n.
    
    # Initial candidates:
    # 1. Small Golomb rulers (Sidon sets)
    # 2. Random sets with varying sizes
    
    # Helper to generate a Golomb ruler
    def greedy_golomb(num_elements):
        U = [0, 1]
        diffs = {1}
        current_max = 1
        for _ in range(num_elements - 2):
            candidate = current_max + 1
            while True:
                valid = True
                for u in U:
                    d = candidate - u
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    U.append(candidate)
                    for u in U[:-1]:
                        diffs.add(candidate - u)
                    current_max = candidate
                    break
                else:
                    candidate += 1
        return U

    # List of sizes to explore
    # Small n: precise search possible
    # Large n: FFT search allows exploring non-Sidon structures
    sizes_to_try = [3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 80]
    
    # We'll keep a list of (score, vec, max_val) for the best found so far
    # But we need to return a list, so we track best_set.
    
    random.seed(42)
    
    # Phase 1: Initialize with Golomb rulers for small sizes
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            U_golomb = greedy_golomb(n)
            max_val = max(U_golomb)
            vec = _list_to_vec(U_golomb, max_val)
            score = _score_from_vector(vec)
            if score > best_score:
                best_score = score
                best_set = U_golomb[:]
        except Exception:
            pass
            
    # Phase 2: Spectral Local Search
    # We will pick a target size n, generate a random set of that size,
    # and then perform local moves (swap, add, remove) to improve the score.
    # To make this efficient, we fix the "container" size M (max value) for a given search block,
    # or we allow M to grow/shrink.
    # Allowing M to change dynamically is complex with FFT (re-padding).
    # Strategy: Fix a maximum bound M_bound for the search.
    # If we find a set with max < M_bound, we can try to "shrink" it by checking smaller M_bounds?
    # Or just keep the vector length fixed to M_bound+1.
    # If we keep length fixed, the score calculation uses the actual max of the non-zero elements.
    
    # Let's define search blocks with different M_bounds.
    # Small M: high resolution, small sets.
    # Large M: can accommodate larger sets or spread out sets.
    
    # Note: The score denominator is log(2*max(U)+1).
    # If we pad to a large M but max(U) is small, the score is calculated correctly based on actual max(U).
    # So we can use a large fixed vector length for FFT efficiency and just track the actual max.
    
    # Let's use a fixed vector length for the FFT to avoid re-allocation overhead?
    # No, FFT cost depends on length. Smaller is faster.
    # Let's group searches by approximate M.
    
    search_configs = [
        (10, 20),   # n in [10, 20], M ~ 100
        (20, 40),   # n in [20, 40], M ~ 200
        (40, 60),   # n in [40, 60], M ~ 400
        (60, 80),   # n in [60, 80], M ~ 600
    ]
    
    # Also try small sets with small M
    small_configs = [
        (3, 5, 50),
        (5, 10, 100),
    ]
    
    all_configs = small_configs + [(n_min, n_max, n_max * n_max * 2) for n_min, n_max in search_configs]
    
    for n_min, n_max, M_bound in all_configs:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Choose a specific n for this block
        # Let's try a few n values within the range
        n_targets = list(range(n_min, n_max + 1, max(1, (n_max - n_min) // 4)))
        
        for n in n_targets:
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Generate initial random set of size n within [0, M_bound]
            # Ensure 0 is included
            U_init = [0] + random.sample(range(1, M_bound + 1), n - 1)
            U_init.sort()
            vec = _list_to_vec(U_init, M_bound)
            
            current_score = _score_from_vector(vec)
            
            # Local search
            # Moves:
            # 1. Swap an element with a random other element (keeping size constant)
            # 2. Move an element to a random new position (keeping size constant)
            # 3. (Optional) Add/remove element if we allow size variation, but let's fix size for now.
            
            # We will use a fixed number of iterations or time-based.
            # Time per config: 2 seconds?
            time_limit = 2.0
            start_time = time.time()
            
            while time.time() < start_time + time_limit:
                if len(U_init) != n: # Safety
                    break
                
                # Pick a move
                move_type = random.random()
                if move_type < 0.7:
                    # Move a random element to a new random position
                    idx = random.randint(0, n - 1)
                    old_val = U_init[idx]
                    # New value
                    new_val = random.randint(0, M_bound)
                    if new_val == old_val:
                        continue
                    # Check if new_val is already in U_init (excluding old_val)
                    # If it is, we would effectively reduce size or need to handle duplicates.
                    # Since U is a set, if new_val is in U_init and new_val != old_val,
                    # we can't just replace. We must pick a new_val not in U_init.
                    if new_val in U_init:
                        continue
                    
                    # Perform move
                    U_init[idx] = new_val
                    U_init.sort()
                    vec = _list_to_vec(U_init, M_bound)
                    new_score = _score_from_vector(vec)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = U_init[:]
                    else:
                        # Revert
                        U_init[idx] = old_val
                        U_init.sort()
                        vec = _list_to_vec(U_init, M_bound)
                else:
                    # Swap two elements? Or just move. Move is sufficient for coverage.
                    # Let's stick to moves to avoid complexity.
                    pass
                    
    # Final check: ensure best_set is valid
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set

# EVOLVE-BLOCK-END
