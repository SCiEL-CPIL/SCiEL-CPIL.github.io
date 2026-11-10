"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        # Check all new sums
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            new_sums.append(s2)
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    return all(v <= 2 for v in counts.values())

def _local_search(chosen, time_limit=10.0):
    """Try swaps and insertions to improve set size."""
    chosen = sorted(chosen)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    start_time = time.time()
    improved = True
    max_iter = 5000
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # First try insertions: add a new element without removing
        chosen_set = set(chosen)
        # Try candidate values in a smart order
        candidates_to_try = []
        for y in range(1, 1001):
            if y not in chosen_set:
                candidates_to_try.append(y)
        
        random.Random().shuffle(candidates_to_try)
        
        for y in candidates_to_try:
            if time.time() - start_time > time_limit:
                break
            # Check if adding y is valid
            ok = True
            for j in range(len(chosen)):
                s = y + chosen[j]
                if counts[s] + 1 > 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * y
                if counts[s2] + 1 > 2:
                    ok = False
            if ok:
                # Do the insertion
                for j in range(len(chosen)):
                    s = y + chosen[j]
                    counts[s] += 1
                counts[2 * y] += 1
                chosen.append(y)
                chosen_set.add(y)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swaps: remove x, add y
        for i in range(len(chosen)):
            if time.time() - start_time > time_limit:
                break
            x = chosen[i]
            # Temporarily remove x
            for j in range(len(chosen)):
                if j != i:
                    s = x + chosen[j]
                    counts[s] -= 1
            counts[2 * x] -= 1
            
            # Try to add some y not in chosen
            chosen_set = set(chosen)
            # Focus on candidates that might fit: try values near existing gaps
            candidate_ys = []
            for y in range(1, 1001):
                if y not in chosen_set:
                    candidate_ys.append(y)
            
            random.Random().shuffle(candidate_ys)
            
            swap_done = False
            for y in candidate_ys:
                if time.time() - start_time > time_limit:
                    break
                # Check if adding y is valid
                ok = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    s = y + chosen[j]
                    if counts[s] + 1 > 2:
                        ok = False
                        break
                if ok:
                    s2 = 2 * y
                    if counts[s2] + 1 > 2:
                        ok = False
                if ok:
                    # Do the swap
                    for j in range(len(chosen)):
                        if j == i:
                            continue
                        s = y + chosen[j]
                        counts[s] += 1
                    counts[2 * y] += 1
                    chosen[i] = y
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    improved = True
                    swap_done = True
                    break
            
            if not swap_done:
                # Restore x
                for j in range(len(chosen)):
                    if j != i:
                        s = x + chosen[j]
                        counts[s] += 1
                counts[2 * x] += 1
    
    return sorted(chosen)

def _structured_greedy(seed_val=0):
    """Try different structured orderings."""
    rng = random.Random(seed_val)
    N = 1000
    
    # Strategy 1: Ascending
    cand1 = list(range(1, N + 1))
    
    # Strategy 2: Descending
    cand2 = list(range(N, 0, -1))
    
    # Strategy 3: Random
    cand3 = list(range(1, N + 1))
    rng.shuffle(cand3)
    
    # Strategy 4: Bias towards smaller numbers
    cand4 = []
    for i in range(1, N + 1):
        # Weight towards smaller numbers
        weight = 1.0 / (1 + i * 0.01)
        cand4.append((weight, i))
    rng.shuffle(cand4)
    cand4 = [x[1] for x in cand4]
    
    # Strategy 5: Sparse then fill
    cand5 = list(range(1, N + 1, 3)) + list(range(2, N + 1, 3))
    
    # Strategy 6: Random with restart bias
    cand6 = list(range(1, N + 1))
    for i in range(N - 1, 0, -1):
        j = rng.randint(0, i)
        cand6[i], cand6[j] = cand6[j], cand6[i]
    
    best = []
    best_size = 0
    
    for cand in [cand1, cand2, cand3, cand4, cand5, cand6]:
        chosen, _ = _greedy_build(cand)
        if len(chosen) > best_size:
            best = chosen
            best_size = len(chosen)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try structured approaches
    if time.time() < deadline - 10.0:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            chosen_s = _structured_greedy(seed)
            if len(chosen_s) > best_size:
                best = chosen_s[:]
                best_size = len(best)
    
    # Strategy 3: Try some random orderings
    rng = random.Random(42)
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        # Create a shuffled candidate list with some structure
        cand = list(range(1, 1001))
        # Mix of random and structured
        if restart % 3 == 0:
            rng.shuffle(cand)
        elif restart % 3 == 1:
            # Bias towards smaller
            weights = [1.0 / (1 + i * 0.02) for i in range(1000)]
            indexed = list(zip(weights, range(1, 1001)))
            rng.shuffle(indexed)
            cand = [x[1] for x in indexed]
        else:
            # Random with some ordering
            half = 500
            first_half = list(range(1, half + 1))
            second_half = list(range(half + 1, 1001))
            rng.shuffle(first_half)
            rng.shuffle(second_half)
            cand = first_half + second_half
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    if time.time() < deadline - 5.0:
        ls_time = min(20.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            best = _local_search(best, time_limit=ls_time)
            best_size = len(best)
    
    # Final verification and cleanup
    if not _is_valid_b2(best):
        # Fallback to standard greedy result
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
