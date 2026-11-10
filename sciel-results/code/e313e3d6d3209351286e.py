"""Rewritten program: randomized construction + local search for covering codes."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

_DEADLINE = 135.0  # safety margin before 150s kill

def _all_words_array(q, n):
    """Return all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words

def _hamming_dists(all_words, center):
    """Compute Hamming distances from center to all words. Returns (N,) array."""
    return np.sum(all_words != center, axis=1)

def _is_covered(all_words, codewords, radius):
    """Check if all words are within radius of some codeword. Returns (N,) bool array of covered."""
    N = len(all_words)
    covered = np.zeros(N, dtype=bool)
    cw = np.array(codewords, dtype=np.int8)
    for i in range(len(cw)):
        dists = _hamming_dists(all_words, cw[i])
        covered[dists <= radius] = True
    return covered

def _greedy_cover_fast(q, n, radius, all_words, time_limit, rng):
    """Fast greedy: pick centers that cover most uncovered words."""
    N = len(all_words)
    uncovered = np.ones(N, dtype=bool)
    codewords = []
    start_time = time.time()
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
            
        # Sample candidates from uncovered words
        n_samples = min(len(uncov_indices), 300)
        sample_idx = rng.choice(uncov_indices, size=n_samples, replace=False)
        
        uncov_words = all_words[uncovered]
        
        best_count = -1
        best_center = None
        
        for ci in range(n_samples):
            center = all_words[sample_idx[ci]]
            dists = np.sum(uncov_words != center, axis=1)
            count = (dists <= radius).sum()
            if count > best_count:
                best_count = count
                best_center = center
        
        if best_center is None:
            break
            
        full_dists = _hamming_dists(all_words, best_center)
        uncovered[full_dists <= radius] = False
        codewords.append(best_center.tolist())
    
    return codewords

def _random_valid_cover(q, n, radius, all_words, time_limit, rng):
    """Build a valid cover by randomly adding codewords until all words are covered."""
    N = len(all_words)
    uncovered = np.ones(N, dtype=bool)
    codewords = []
    start_time = time.time()
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random uncovered word as center
        uncov_indices = np.where(uncovered)[0]
        idx = rng.choice(uncov_indices)
        center = all_words[idx]
        
        dists = _hamming_dists(all_words, center)
        uncovered[dists <= radius] = False
        codewords.append(center.tolist())
    
    return codewords

def _local_improve(codewords, q, n, radius, all_words, time_limit, rng):
    """Local search: try to remove redundant codewords or swap to enable removals."""
    N = len(all_words)
    codewords = [list(cw) for cw in codewords]
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        cw_arr = np.array(codewords, dtype=np.int8)
        k = len(codewords)
        
        # Try removing each codeword
        for i in range(k):
            if time.time() - start_time > time_limit:
                break
            remaining = list(codewords[:i]) + list(codewords[i+1:])
            if not remaining:
                continue
            rem_arr = np.array(remaining, dtype=np.int8)
            # Check if all words are still covered
            covered = np.zeros(N, dtype=bool)
            for j in range(len(rem_arr)):
                dists = _hamming_dists(all_words, rem_arr[j])
                covered[dists <= radius] = True
            if covered.all():
                codewords = remaining
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping a codeword to a random position and see if we can remove another
        if len(codewords) > 1:
            for i in range(len(codewords)):
                if time.time() - start_time > time_limit:
                    break
                # Try moving codeword i to a random new position
                new_pos = rng.randint(0, q-1)
                pos_idx = rng.randint(0, n-1)
                old_val = codewords[i][pos_idx]
                if old_val == new_pos:
                    continue
                codewords[i][pos_idx] = new_pos
                
                # Now try to remove some other codeword
                k = len(codewords)
                found_removal = False
                for j in range(k):
                    if j == i:
                        continue
                    remaining = list(codewords[:j]) + list(codewords[j+1:])
                    if not remaining:
                        continue
                    rem_arr = np.array(remaining, dtype=np.int8)
                    covered = np.zeros(N, dtype=bool)
                    for m in range(len(rem_arr)):
                        dists = _hamming_dists(all_words, rem_arr[m])
                        covered[dists <= radius] = True
                    if covered.all():
                        codewords = remaining
                        improved = True
                        found_removal = True
                        break
                
                if not found_removal:
                    # Revert the swap
                    codewords[i][pos_idx] = old_val
    
    return codewords

def _verify_coverage(codewords, q, n, radius, all_words):
    """Check that every word is within radius of some codeword."""
    N = len(all_words)
    covered = np.zeros(N, dtype=bool)
    cw = np.array(codewords, dtype=np.int8)
    for i in range(len(cw)):
        dists = _hamming_dists(all_words, cw[i])
        covered[dists <= radius] = True
    return covered.all()

def construct(q, n, radius):
    start_time = time.time()
    
    # Generate all words
    all_words = _all_words_array(q, n)
    N = len(all_words)
    
    best_result = None
    best_len = float('inf')
    
    # Phase 1: Greedy cover
    time_remaining = _DEADLINE - (time.time() - start_time)
    if time_remaining > 10:
        rng = random.Random(42)
        greedy_time = min(30.0, time_remaining * 0.3)
        result = _greedy_cover_fast(q, n, radius, all_words, greedy_time, rng)
        if result and _verify_coverage(result, q, n, radius, all_words):
            best_len = len(result)
            best_result = result
    
    # Phase 2: Local improvement on greedy result
    if best_result is not None:
        time_remaining = _DEADLINE - (time.time() - start_time)
        if time_remaining > 15:
            rng = random.Random(123)
            improve_time = min(40.0, time_remaining * 0.4)
            improved = _local_improve(best_result, q, n, radius, all_words, improve_time, rng)
            if improved and _verify_coverage(improved, q, n, radius, all_words):
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_result = improved
    
    # Phase 3: Random valid cover + local improvement (different starting point)
    time_remaining = _DEADLINE - (time.time() - start_time)
    if time_remaining > 20:
        rng = random.Random(777)
        rand_time = min(15.0, time_remaining * 0.2)
        rand_result = _random_valid_cover(q, n, radius, all_words, rand_time, rng)
        if rand_result and _verify_coverage(rand_result, q, n, radius, all_words):
            time_remaining = _DEADLINE - (time.time() - start_time)
            if time_remaining > 10:
                improve_time = min(30.0, time_remaining * 0.5)
                improved = _local_improve(rand_result, q, n, radius, all_words, improve_time, rng)
                if improved and _verify_coverage(improved, q, n, radius, all_words):
                    if len(improved) < best_len:
                        best_len = len(improved)
                        best_result = improved
    
    # Phase 4: More local improvement on best result
    if best_result is not None:
        time_remaining = _DEADLINE - (time.time() - start_time)
        if time_remaining > 10:
            rng = random.Random(999)
            improve_time = min(25.0, time_remaining * 0.6)
            improved = _local_improve(best_result, q, n, radius, all_words, improve_time, rng)
            if improved and _verify_coverage(improved, q, n, radius, all_words):
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_result = improved
    
    if best_result is not None:
        return best_result
    
    # Fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
