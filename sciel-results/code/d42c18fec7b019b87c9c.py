import time
import random
import numpy as np
from itertools import product
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute balls (index lists) for each word
    # Ball(w) contains w and all words at distance 1
    # We need a fast way to check coverage.
    # We will use an array `cover_count` of size N.
    
    # Precompute neighbor lists for fast updates
    # neighbors[i] = list of indices covered by word i (including i)
    neighbors = []
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
    
    # Convert neighbors to numpy arrays for faster indexing? 
    # Or just keep as lists. N=2187 is small.
    
    # 1. Initial Construction: Greedy Independent Set
    # We want a set of codewords such that no two are at distance <= 2.
    # This ensures that their balls (radius 1) are disjoint? No.
    # If dist(c1, c2) >= 3, then Ball(c1) and Ball(c2) are disjoint?
    # dist(c1, c2) = d. 
    # If d=3, there is no word at dist 1 from c1 and dist 1 from c2.
    # Proof: If x is dist 1 from c1 and dist 1 from c2, then dist(c1, c2) <= dist(c1, x) + dist(x, c2) = 1 + 1 = 2.
    # So yes, if pairwise distance >= 3, the balls are disjoint.
    # Thus, a code with min distance 3 is a packing, not necessarily a covering.
    # But it's a good starting point because it has no redundancy (no two codewords cover the same point).
    # Wait, if balls are disjoint, then cover_count is either 0 or 1 for all points in the union.
    # We need to cover the *entire* space.
    # A maximal independent set in the distance-2 graph is a code with min distance >= 3 that is maximal.
    # It might not cover the space.
    # So we start with a greedy independent set, then we MUST add codewords to cover the rest.
    # This "add to cover" step will introduce overlaps.
    
    # Alternative Initialization: Greedy Set Cover (as before) but optimized.
    # Let's stick to Greedy Set Cover for initialization to ensure validity immediately.
    # Then use Local Search to remove redundancies.
    
    covered = np.zeros(N, dtype=bool)
    codewords = []
    remaining = set(range(N))
    
    # To speed up greedy, we can use a heuristic: pick the word that covers the most *uncovered* points.
    # We maintain a count of uncovered points for each candidate? 
    # That's expensive to update.
    # Since N is small, we can just scan.
    
    # Optimization: Precompute bitsets? 
    # 2187 bits = 35 uint64s.
    # Let's use Python integers as bitsets (arbitrary precision).
    # This is very fast in Python.
    
    # Precompute bitsets for each ball
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    
    # Greedy Set Cover
    # We need to find the word that adds the most new bits to current_cover_mask.
    # new_bits = ball_bitsets[i] & ~current_cover_mask
    # popcount(new_bits)
    
    # To make this fast, we can just iterate. 
    # 2187 candidates * ~186 iterations = 400k iterations.
    # Each iteration does a big int AND and popcount. Python big ints are fast.
    
    while current_cover_mask != all_mask:
        if time.time() > deadline - 60:
            break # Safety break
            
        best_i = -1
        best_count = 0
        
        # Sample a subset of candidates if too slow? 
        # With N=2187, scanning all is likely fast enough in Python if we use bit ops.
        # Let's try scanning all.
        
        for i in range(N):
            if i in codewords: # Optimization: skip already chosen? 
                # Actually, re-adding is useless. 
                # But checking `i in codewords` is O(L). 
                # Better: maintain a boolean array `is_chosen`.
                pass
                
        # Let's use a boolean array for chosen status to skip them in the loop?
        # Or just compute.
        
        # Actually, we can just scan all.
        for i in range(N):
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count() # Python 3.8+
                if count > best_count:
                    best_count = count
                    best_i = i
                    
        if best_i == -1:
            break # Should not happen if mask != all_mask
            
        codewords.append(best_i)
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure we have a valid cover. If we broke early, finish it.
    if current_cover_mask != all_mask:
        # Fallback: add random words until covered
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            current_cover_mask |= ball_bitsets[i]
            if i not in codewords:
                codewords.append(i)
            if time.time() > deadline - 10:
                break

    # 2. Local Search
    # We have a valid cover. We want to minimize len(codewords).
    # State: codewords (list), cover_count (np array of int8/int16)
    
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        for j in neighbors[cw]:
            cover_count[j] += 1
    
    # Precompute neighbor sets for fast membership testing
    neighbor_sets = [set(nb) for nb in neighbors]
    
    def try_remove(cw_idx):
        b = neighbors[cw_idx]
        if np.min(cover_count[b]) > 1:
            cover_count[b] -= 1
            return True
        return False

    # Redundancy Removal Pass
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 30:
            break

    # Local Search Loop
    # Strategy:
    # 1. Try to remove a random codeword.
    # 2. If fails, try 1-for-1 swap.
    # 3. If fails, try 2-for-1 swap (remove 2, add 1).
    
    random.seed(42)
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Removal
        if codewords:
            idx = random.randint(0, len(codewords) - 1)
            cw = codewords[idx]
            if try_remove(cw):
                codewords.pop(idx)
                improved = True
                continue
        
        # 2. Try 1-for-1 Swap
        if len(codewords) >= 1:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick random cw_new
            cw_new = random.randint(0, N - 1)
            if cw_new == cw_old:
                continue
            
            b_old = neighbors[cw_old]
            b_new = neighbors[cw_new]
            mask_old = ball_bitsets[cw_old]
            mask_new = ball_bitsets[cw_new]
            
            # Check if b_old subset of b_new
            if (mask_old & ~mask_new) == 0:
                cover_count[b_old] -= 1
                cover_count[b_new] += 1
                codewords[idx_old] = cw_new
                improved = True
                continue
            
            # Otherwise, check specific points
            cover_count[b_old] -= 1
            cover_count[b_new] += 1
            
            if np.min(cover_count) > 0:
                codewords[idx_old] = cw_new
                improved = True
            else:
                cover_count[b_old] += 1
                cover_count[b_new] -= 1
        
        # 3. Try 2-for-1 Swap: remove 2 codewords, add 1
        if len(codewords) >= 2:
            idx1 = random.randint(0, len(codewords) - 1)
            idx2 = random.randint(0, len(codewords) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(codewords) - 1)
            
            cw1 = codewords[idx1]
            cw2 = codewords[idx2]
            
            # Pick random cw_new
            cw_new = random.randint(0, N - 1)
            if cw_new == cw1 or cw_new == cw2:
                continue
            
            b1 = neighbors[cw1]
            b2 = neighbors[cw2]
            b_new = neighbors[cw_new]
            
            # Tentatively remove both, add new
            cover_count[b1] -= 1
            cover_count[b2] -= 1
            cover_count[b_new] += 1
            
            if np.min(cover_count) > 0:
                # Remove the two codewords (handle index adjustment)
                if idx1 < idx2:
                    codewords.pop(idx1)
                    codewords.pop(idx2 - 1)
                else:
                    codewords.pop(idx2)
                    codewords.pop(idx1 - 1)
                codewords.append(cw_new)
                improved = True
            else:
                # Revert
                cover_count[b1] += 1
                cover_count[b2] += 1
                cover_count[b_new] -= 1

    # Final redundancy removal pass
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
    
    # Final Result
    result = [list(all_words[cw]) for cw in codewords]
    return result
