"""Combined program for ternary covering code K_3(7,1) using optimized greedy set cover + fast local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a greedy set-cover approach followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    # Fallback: direct sum construction for general case
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    """Direct sum construction as fallback."""
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    # For q=3, use tetracode-based construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    """Solve K_3(7,1) using greedy set cover + fast local search with bitsets."""
    q, n, radius = 3, 7, 1
    
    # 1. Precompute all words and their balls (neighbors within radius 1)
    all_words = list(product(range(3), repeat=7))
    N = len(all_words) # 2187
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as bitsets (Python integers) for fast operations
    # N=2187 fits easily in a Python int (arbitrary precision)
    balls_bits = [0] * N
    for w in all_words:
        idx = word_to_idx[w]
        bits = 1 << idx
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    new_word = tuple(new_word)
                    bits |= (1 << word_to_idx[new_word])
        balls_bits[idx] = bits
    
    # Also keep numpy arrays for coverage count updates
    ball_arrays = []
    for i in range(N):
        # Extract indices from bits
        b = balls_bits[i]
        indices = []
        while b:
            lsb = b & (-b)
            idx = lsb.bit_length() - 1
            indices.append(idx)
            b ^= lsb
        ball_arrays.append(np.array(indices, dtype=np.int32))
    
    # 2. Greedy Set Cover to get initial solution
    covered_bits = 0
    codewords = []
    
    # For greedy, we need to find the word that covers the most uncovered points
    # We can use bit operations: uncovered = all_mask ^ covered_bits
    # gain = popcount(balls_bits[i] & uncovered)
    # But popcount in Python is fast (int.bit_count())
    
    all_mask = (1 << N) - 1
    
    while covered_bits != all_mask:
        uncovered = all_mask ^ covered_bits
        best_count = 0
        best_cw = -1
        for i in range(N):
            gain = (balls_bits[i] & uncovered).bit_count()
            if gain > best_count:
                best_count = gain
                best_cw = i
                if best_count == 16:  # Max possible gain
                    break
        if best_cw == -1 or best_count == 0:
            break
        codewords.append(best_cw)
        covered_bits |= balls_bits[best_cw]
        
        if time.time() > deadline - 30:
            break

    if covered_bits != all_mask:
        # Fallback to finish quickly
        while covered_bits != all_mask:
            uncovered = all_mask ^ covered_bits
            if uncovered == 0:
                break
            # Pick a random uncovered point
            lsb = uncovered & (-uncovered)
            idx = lsb.bit_length() - 1
            # Find a codeword covering this point
            # Just use the point itself as codeword
            codewords.append(idx)
            covered_bits |= balls_bits[idx]
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[ball_arrays[cw]] += 1
        
    # Helper to remove codeword
    def try_remove(cw_idx):
        b = ball_arrays[cw_idx]
        # Check if any word in b has count 1
        if np.min(coverage_count[b]) == 1:
            return False
        # Safe to remove
        coverage_count[b] -= 1
        return True

    # 4. Redundancy Removal (Greedy Pass)
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 20:
            break

    # 5. Local Search with bitset-based swap check
    random.seed(42)
    
    # Precompute for each pair of codewords the difference bits (Ball(A) \ Ball(B))
    # But that's N^2 which is too much. Instead, compute on the fly using bit operations.
    
    while time.time() < deadline - 2:
        # Try to remove a random codeword
        if codewords:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                continue
            
        # Try to swap
        if codewords:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword
            cw_new = random.randint(0, N - 1)
            
            if cw_new == cw_old:
                continue
            
            # Check if swap is valid using bitsets
            # Points lost: Ball(cw_old) & ~Ball(cw_new)
            lost_bits = balls_bits[cw_old] & ~balls_bits[cw_new]
            
            if lost_bits == 0:
                # No points lost, swap is valid
                codewords[idx_old] = cw_new
                coverage_count[ball_arrays[cw_old]] -= 1
                coverage_count[ball_arrays[cw_new]] += 1
                continue
            
            # Check if any lost point has coverage count == 1
            # We need to check coverage_count for all indices in lost_bits
            # Extract indices from lost_bits
            lost_indices = []
            temp = lost_bits
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                lost_indices.append(idx)
                temp ^= lsb
            
            if len(lost_indices) == 0:
                # Shouldn't happen if lost_bits != 0
                codewords[idx_old] = cw_new
                coverage_count[ball_arrays[cw_old]] -= 1
                coverage_count[ball_arrays[cw_new]] += 1
                continue
            
            # Check minimum coverage among lost points
            if np.min(coverage_count[lost_indices]) == 1:
                # Invalid swap
                continue
            else:
                # Valid swap
                codewords[idx_old] = cw_new
                coverage_count[ball_arrays[cw_old]] -= 1
                coverage_count[ball_arrays[cw_new]] += 1

    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
