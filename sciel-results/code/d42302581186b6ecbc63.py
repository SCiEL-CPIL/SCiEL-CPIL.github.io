"""Ternary covering code K_3(7,1) via simulated annealing on selection vector."""
import time
import numpy as np
import random

_START = time.time()
_DEADLINE = _START + 110.0

def _time_ok():
    return time.time() < _DEADLINE

def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 110.0

    if radius != 1:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n
    N = total
    
    # Precompute neighbors: neighbors[i] is a list of indices within dist 1 of i
    # For q=3, n=7, each ball has 1 + 7*(3-1) = 15 elements
    neighbors = np.empty((N, 15), dtype=np.int32)
    for i in range(N):
        word = np.unravel_index(i, (q,) * n)
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            orig_val = word[pos]
            for v in range(q):
                if v == orig_val:
                    continue
                new_word = list(word)
                new_word[pos] = v
                neighbors[i, idx] = np.ravel_multi_index(tuple(new_word), (q,) * n)
                idx += 1
    # idx should be 15

    # Precompute transpose-like structure for faster energy calculation?
    # Energy = uncovered_count * 100 + sum(x)
    # uncovered_count = sum( (1 - coverage) )
    # coverage[j] = 1 if any x[i]=1 and j in neighbors[i]
    
    # To speed up updates:
    # When we toggle x[i] (0->1 or 1->0), coverage changes for neighbors[i].
    # We maintain `cover_count[j]` = number of selected codewords covering j.
    # uncovered = count of j where cover_count[j] == 0.
    # code_size = sum(x).
    
    best_solution = None
    best_energy = float('inf')
    
    # Multi-restart
    while _time_ok():
        # Initialization: Random subset with expected size ~ 180
        # Or start with a greedy solution? Greedy is fast to compute once.
        # Let's use a quick greedy init to get a good starting point for SA.
        
        # Quick Greedy Init
        covered = np.zeros(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        remaining = set(range(N))
        # To make greedy fast, we can just pick random words until covered, or simple greedy.
        # Simple greedy is O(N^2 * ball_size) which is 2187^2 * 15 ~ 70M ops. Might be slow in Python loop.
        # Let's use a randomized heuristic init: pick random words, check coverage.
        # Or just start SA from a random sparse vector.
        
        # Better: Start from a known decent structure or random.
        # Let's try random init with p=0.08 (expected 175 words)
        x = (np.random.rand(N) < 0.08).astype(np.int8)
        
        # Calculate initial cover_count and energy
        cover_count = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if x[i]:
                cover_count[neighbors[i]] += 1
        
        uncovered = np.sum(cover_count == 0)
        code_size = np.sum(x)
        energy = uncovered * 100 + code_size
        
        if energy < best_energy:
            best_energy = energy
            best_solution = x.copy()
            
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        current_energy = energy
        current_cover_count = cover_count.copy()
        current_uncovered = uncovered
        current_size = code_size
        
        # Annealing loop
        while T > T_min and _time_ok():
            # Pick random index to flip
            i = np.random.randint(N)
            
            # Calculate delta energy
            # If x[i] is 0 -> 1:
            if x[i] == 0:
                # New words covered: those in neighbors[i] with cover_count == 0
                # Note: neighbors[i] contains i itself.
                new_covered = np.sum(current_cover_count[neighbors[i]] == 0)
                delta_uncovered = -new_covered
                delta_size = 1
            else:
                # x[i] is 1 -> 0:
                # Words that might become uncovered: those in neighbors[i] with cover_count == 1
                # AND not covered by any other selected word.
                # Since we are removing i, cover_count[j] for j in neighbors[i] decreases by 1.
                # If cover_count[j] was 1, it becomes 0.
                # So new_uncovered = count of j in neighbors[i] where cover_count[j] == 1
                new_uncovered = np.sum(current_cover_count[neighbors[i]] == 1)
                delta_uncovered = new_uncovered
                delta_size = -1
            
            delta_energy = delta_uncovered * 100 + delta_size
            
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / T):
                # Accept move
                x[i] = 1 - x[i]
                
                # Update cover_count
                if x[i] == 1:
                    current_cover_count[neighbors[i]] += 1
                else:
                    current_cover_count[neighbors[i]] -= 1
                
                current_uncovered += delta_uncovered
                current_size += delta_size
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = x.copy()
            
            T *= cooling
            
            # Periodic check for time
            if not _time_ok():
                break

    # Final verification and cleanup
    if best_solution is None:
        # Fallback
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    # Extract indices
    indices = np.nonzero(best_solution)[0]
    
    # Verify validity
    cover_check = np.zeros(N, dtype=bool)
    for i in indices:
        cover_check[neighbors[i]] = True
    
    if not cover_check.all():
        # If invalid, try to fix by adding missing words (shouldn't happen if energy penalty worked, but safe)
        missing = np.nonzero(cover_check == False)[0]
        for m in missing:
            # Find a word covering m
            # Just add m itself
            indices = np.append(indices, m)
            cover_check[neighbors[m]] = True
            
    # Convert to list of lists
    out = []
    for i in indices:
        out.append(list(np.unravel_index(i, (q,) * n)))
        
    return out
