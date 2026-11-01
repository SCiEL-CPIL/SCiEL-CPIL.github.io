"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases or if optimization fails
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for 120s limit
    
    # Precompute neighbors and cover bits
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    neighbors = []
    cover_bits = []
    
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        bit = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nb_idx = word_to_idx(w_copy)
                nbs.append(nb_idx)
                bit |= (1 << nb_idx)
        neighbors.append(nbs)
        cover_bits.append(bit)
        
    # Convert neighbors to numpy arrays for fast min operations
    neighbors_arr = np.array(neighbors, dtype=int)
    
    def greedy_cover(randomize=False):
        uncovered = (1 << total_words) - 1
        code = []
        remaining = list(range(total_words))
        while uncovered:
            best_idx = -1
            max_cov = -1
            # If randomize, we can shuffle remaining or pick randomly from top-k
            # For speed, we just iterate.
            for c in remaining:
                cov = (uncovered & cover_bits[c]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = c
                    if randomize and max_cov > 0:
                        # Optional: break early if we find a good one?
                        # No, greedy needs best.
                        pass
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            uncovered &= ~cover_bits[best_idx]
            remaining.remove(best_idx)
            
        return code

    def local_search(code):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pass 1: Removals
            for c in list(code_set):
                if cover_count[neighbors_arr[c]].min() > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    improved = True
            
            if improved:
                continue
                
            # Pass 2: Swaps
            # Try to find a valid swap to perturb the state
            # We limit attempts to avoid wasting time on dead ends
            for _ in range(50):
                if not code_set:
                    break
                c = random.choice(list(code_set))
                # Get critical words (covered only by c)
                crit = [w for w in neighbors[c] if cover_count[w] == 1]
                if not crit:
                    # Should have been removed, but maybe due to order?
                    # Force removal
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    break
                    
                # Try random x
                x = random.randint(0, total_words - 1)
                if x in code_set:
                    continue
                    
                # Check if x covers all critical words of c
                valid = True
                for w in crit:
                    if not (cover_bits[x] >> w) & 1:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    cover_count[neighbors_arr[c]] -= 1
                    cover_count[neighbors_arr[x]] += 1
                    code_set.remove(c)
                    code_set.add(x)
                    improved = True
                    break
                    
        return list(code_set)

    best_code = None
    
    # Run multiple rounds
    # First, a deterministic greedy
    cand = greedy_cover(randomize=False)
    cand = local_search(cand)
    best_code = cand
    
    # Then, randomized restarts
    while time.time() < deadline:
        # Randomized greedy
        # To make it different, we can shuffle the order of evaluation in greedy
        # Or just run local search on the same start? No, need diversity.
        # Let's implement a slightly randomized greedy
        def greedy_cover_random():
            uncovered = (1 << total_words) - 1
            code = []
            remaining = list(range(total_words))
            while uncovered:
                # Pick a random subset of candidates? Or just pick best with some noise?
                # Simple: Pick best, but if multiple have same score, pick random.
                best_idx = -1
                max_cov = -1
                candidates = []
                for c in remaining:
                    cov = (uncovered & cover_bits[c]).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        candidates = [c]
                    elif cov == max_cov:
                        candidates.append(c)
                        
                if not candidates:
                    break
                    
                best_idx = random.choice(candidates)
                code.append(best_idx)
                uncovered &= ~cover_bits[best_idx]
                remaining.remove(best_idx)
            return code

        cand = greedy_cover_random()
        cand = local_search(cand)
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = local_search(best_code)
        
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
