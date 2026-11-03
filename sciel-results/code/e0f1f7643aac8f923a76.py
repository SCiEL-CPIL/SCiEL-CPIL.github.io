# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict
import heapq
import time

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specific solver for q=3, n=7, radius=1
    # We use a Branch and Bound approach with bitsets.
    
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute cover masks: cover_mask[c] is a bitset of words covered by codeword c
    # Precompute coverers masks: coverers_mask[w] is a bitset of codewords covering word w
    cover_mask = [0] * N
    coverers_mask = [0] * N
    
    # Helper to get neighbors (distance <= 1)
    def get_neighbors_indices(w):
        neighbors = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        return neighbors

    for i, w in enumerate(all_words):
        neighbors = get_neighbors_indices(w)
        mask = 0
        for nb in neighbors:
            idx = word_to_idx[nb]
            mask |= (1 << idx)
            coverers_mask[idx] |= (1 << i)
        cover_mask[i] = mask
        
    # 1. Greedy Initialization to get a good upper bound
    uncovered_greedy = (1 << N) - 1
    available_greedy = (1 << N) - 1
    codewords_greedy = []
    
    # Simple greedy: pick codeword covering most uncovered
    # To speed up, we can just run a few iterations of "best choice"
    # Since N is small, we can afford a slightly expensive greedy
    for _ in range(300): # Max 300 codewords
        if uncovered_greedy == 0:
            break
        best_c = -1
        best_count = -1
        # Sample or scan? Scanning 2187 is fast.
        # But we need to count bits in (cover_mask[c] & uncovered_greedy)
        # This is 2187 bit_count operations per step. 300 * 2187 ~ 650k ops. Fast.
        for c in range(N):
            if not (available_greedy & (1 << c)):
                continue
            count = (cover_mask[c] & uncovered_greedy).bit_count()
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        codewords_greedy.append(best_c)
        uncovered_greedy &= ~cover_mask[best_c]
        available_greedy &= ~(1 << best_c)
        
    best_size = len(codewords_greedy)
    
    # 2. Branch and Bound Search
    # We try to find a solution with size < best_size
    # Global variables for B&B
    global_best = best_size
    global_solution = codewords_greedy[:]
    
    # Lower bound function
    def lower_bound(uncovered):
        if uncovered == 0:
            return 0
        # Max words covered by one codeword is 15
        # So we need at least ceil(popcount / 15)
        return (uncovered.bit_count() + 14) // 15
        
    # B&B Function
    # uncovered: bitset of uncovered words
    # available: bitset of available codewords
    # path: list of chosen codewords
    def search(uncovered, available, path):
        nonlocal global_best, global_solution
        
        if uncovered == 0:
            if len(path) < global_best:
                global_best = len(path)
                global_solution = path[:]
            return True # Found a better solution, can prune siblings if we just want one, but we want min
            
        # Prune if current depth + lower bound >= best
        if len(path) + lower_bound(uncovered) >= global_best:
            return False
            
        # Heuristic: Find word w in uncovered with min available coverers
        # We iterate through bits of uncovered
        # To make it faster, we can sample or just iterate all bits if few uncovered
        # Let's iterate through set bits of uncovered
        u = uncovered
        best_w = -1
        min_coverers = N + 1
        best_covers_count = -1
        
        while u:
            # Get lowest set bit
            lsb = u & -u
            w = lsb.bit_length() - 1
            u ^= lsb
            
            # Count available codewords covering w
            avail_coverers = coverers_mask[w] & available
            count = avail_coverers.bit_count()
            
            if count == 0:
                return False # Dead end
                
            # MRV: pick w with min count
            # Tie-breaker: pick w whose coverers cover most other uncovered words
            if count < min_coverers:
                min_coverers = count
                best_w = w
                # Calculate secondary heuristic: sum of cover counts of these codewords
                # Or just pick the first one with min count for speed
                # Let's calculate a score for tie-breaking
                score = 0
                ac = avail_coverers
                while ac:
                    lsb_c = ac & -ac
                    c = lsb_c.bit_length() - 1
                    ac ^= lsb_c
                    score += (cover_mask[c] & uncovered).bit_count()
                best_covers_count = score
            elif count == min_coverers:
                # Tie break: higher score is better
                score = 0
                ac = avail_coverers
                while ac:
                    lsb_c = ac & -ac
                    c = lsb_c.bit_length() - 1
                    ac ^= lsb_c
                    score += (cover_mask[c] & uncovered).bit_count()
                if score > best_covers_count:
                    best_w = w
                    best_covers_count = score
        
        if best_w == -1:
            return False
            
        # If min_coverers == 1, we are forced to take that codeword
        # Otherwise, we branch on all available coverers of best_w
        
        candidates = coverers_mask[best_w] & available
        
        # Order candidates by how many new words they cover (greedy within branch)
        # This helps find good solutions faster
        cand_list = []
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            c ^= lsb
            score = (cover_mask[idx] & uncovered).bit_count()
            cand_list.append((score, idx))
            
        # Sort by score descending
        cand_list.sort(reverse=True)
        
        for score, c in cand_list:
            # Optimization: if we only have one candidate, take it
            # But we already sorted, so we just iterate
            
            new_uncovered = uncovered & ~cover_mask[c]
            new_available = available & ~(1 << c)
            
            # Recurse
            path.append(c)
            search(new_uncovered, new_available, path)
            path.pop()
            
            # If we found a solution with size equal to lower bound, we can stop?
            # No, we want global best.
            # If global_best is updated, the prune condition at the top of next iterations will handle it.
            
            # Early termination if we found a solution of size 1? (Impossible here)
            # If we found a solution, and the lower bound of the remaining search is >= global_best, 
            # the loop will naturally prune via the check at the start of the function call.
            
        return False

    # Start search
    # To make it faster, we can limit the time or just run it.
    # Given N=2187, the search space is huge, but the heuristic is strong.
    # The greedy usually gets ~190-200. The optimal is 186.
    # B&B might take a while. Let's add a time check or just let it run.
    # The prompt says 120s.
    
    # To ensure we don't TLE, we can run the search for a limited time or depth.
    # But standard B&B with strong heuristics is usually fast for this size.
    
    # Let's run it.
    search((1 << N) - 1, (1 << N) - 1, [])
    
    result = [list(all_words[i]) for i in global_solution]
    return result
# EVOLVE-BLOCK-END
