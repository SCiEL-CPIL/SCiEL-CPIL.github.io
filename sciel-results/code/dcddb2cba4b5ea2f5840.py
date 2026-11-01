import time
import random
import numpy as np
from itertools import product

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
            
    def is_valid():
        return np.min(cover_count) > 0

    def try_remove(cw_idx):
        # Check if all points in ball(cw_idx) have count > 1
        # If yes, decrement and return True
        b = neighbors[cw_idx]
        # Use numpy for speed
        if np.min(cover_count[b]) > 1:
            cover_count[b] -= 1
            return True
        return False

    def try_add(cw_idx):
        b = neighbors[cw_idx]
        cover_count[b] += 1
        return True

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
    # 2. If fails, try to swap it with a candidate that covers the "lost" points.
    
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
        
        # 2. Try Swap
        if codewords:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Tentatively remove cw_old
            b_old = neighbors[cw_old]
            cover_count[b_old] -= 1
            
            # Find points that became 0
            # We need to find a cw_new such that Ball(cw_new) covers all these points.
            # Let U = {j | cover_count[j] == 0}
            # We need cw_new such that U is subset of Ball(cw_new).
            
            # To find such cw_new efficiently:
            # We can precompute for each point j, the list of codewords covering it.
            # But the set of "good" cw_new depends on U.
            
            # Heuristic: Pick a random point u from U.
            # Look for candidates covering u.
            # Check if they cover all of U.
            
            # First, identify U.
            # Since N is small, we can just scan cover_count.
            # But doing this for every swap is expensive.
            # Optimization: Only consider swaps if the "cost" of removal is small.
            
            # Let's just revert and do a simpler heuristic:
            # Randomly pick a new codeword cw_new.
            # Check if swap is valid.
            
            # Revert
            cover_count[b_old] += 1
            
            # Pick random cw_new
            cw_new = random.randint(0, N - 1)
            if cw_new == cw_old:
                continue
            
            b_new = neighbors[cw_new]
            
            # Check validity of swap
            # We need min(cover_count - cover_count[b_old] + cover_count[b_new]) > 0
            # This is hard to vectorize directly without creating a temp array.
            # Temp array of size 2187 is fast.
            
            # Create a view? No, just compute.
            # diff = cover_count.copy()
            # diff[b_old] -= 1
            # diff[b_new] += 1
            # if np.min(diff) > 0: accept
            
            # To save time, check only the union of b_old and b_new?
            # Points outside b_old and b_new are unchanged.
            # Points in b_old \ b_new: count decreases by 1. Must be > 1 before dec (i.e. > 0 after).
            # Points in b_new \ b_old: count increases by 1. Always ok.
            # Points in b_old \cap b_new: count unchanged.
            
            # So we only need to check points in b_old that are NOT in b_new.
            # And points in b_old that ARE in b_new are fine (count unchanged).
            # Wait, if a point is in b_old, its count drops by 1.
            # If it is ALSO in b_new, its count rises by 1. Net 0.
            # If it is NOT in b_new, its count drops by 1.
            
            # So we need to check: for all j in b_old \ b_new, cover_count[j] > 1.
            
            # How to find b_old \ b_new efficiently?
            # We can use sets? No, numpy.
            # We can precompute bitsets.
            
            # Let's use the bitsets again.
            # mask_old = ball_bitsets[cw_old]
            # mask_new = ball_bitsets[cw_new]
            # mask_diff = mask_old & ~mask_new
            
            # If mask_diff == 0, swap is always valid (new covers all old's unique points? No, new covers all old's points? No.)
            # If mask_diff == 0, it means b_old is subset of b_new.
            # Then all points in b_old are also in b_new.
            # For j in b_old: count(j) - 1 (from old) + 1 (from new) = count(j). Unchanged.
            # So if b_old subset of b_new, swap is ALWAYS valid.
            
            # If b_old is NOT subset of b_new, we have points in b_old \ b_new.
            # Their count drops by 1. We need them to be > 1.
            
            mask_old = ball_bitsets[cw_old]
            mask_new = ball_bitsets[cw_new]
            
            # Check if b_old subset of b_new
            # (mask_old & ~mask_new) == 0
            if (mask_old & ~mask_new) == 0:
                # Valid swap
                cover_count[b_old] -= 1
                cover_count[b_new] += 1
                codewords[idx_old] = cw_new
                improved = True
                continue
            
            # Otherwise, check the specific points
            # Get indices of points in b_old \ b_new
            # This requires converting bitset to indices, which is slow.
            # Alternative: Just iterate through b_old (list of ~15 items) and check if they are in b_new.
            # b_new is a list of ~15 items.
            # We can make a set of b_new for O(1) lookup?
            # Or just check cover_count[j] > 1 for j in b_old, 
            # but we must ensure we only check those NOT covered by new.
            
            # Actually, we can just do the full check on the small union.
            # Union size is at most 30.
            # We can create a small numpy array of the union indices?
            
            # Simpler:
            # 1. Decrement b_old
            # 2. Increment b_new
            # 3. Check min(cover_count) > 0
            # 4. If fail, revert.
            
            # This is O(N) for the min check? No, min is O(N).
            # N=2187. 2187 ops per swap.
            # 1000 swaps = 2M ops. Fast.
            
            cover_count[b_old] -= 1
            cover_count[b_new] += 1
            
            if np.min(cover_count) > 0:
                codewords[idx_old] = cw_new
                improved = True
            else:
                # Revert
                cover_count[b_old] += 1
                cover_count[b_new] -= 1

    # Final Result
    result = [list(all_words[cw]) for cw in codewords]
    return result
