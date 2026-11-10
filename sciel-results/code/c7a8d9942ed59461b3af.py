"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    for x in candidates:
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
    """Try swaps to improve set size."""
    chosen = sorted(chosen)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try to find a swap: remove x, add y not in set
        for i in range(len(chosen)):
            if time.time() - start_time > time_limit:
                break
            x = chosen[i]
            # Temporarily remove x
            for j in range(len(chosen)):
                if j == i:
                    s = x + chosen[j]
                    counts[s] -= 1
            counts[2 * x] -= 1
            # Try to add some y not in chosen
            chosen_set = set(chosen)
            # Candidate y values: try values that might fit
            for y in range(1, 1001):
                if y in chosen_set:
                    continue
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
                    # Do the swap
                    for j in range(len(chosen)):
                        s = y + chosen[j]
                        counts[s] += 1
                    counts[2 * y] += 1
                    chosen[i] = y
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    improved = True
                    break
            if improved:
                break
            else:
                # Restore x
                for j in range(len(chosen)):
                    if j != i:
                        s = x + chosen[j]
                        counts[s] += 1
                counts[2 * x] += 1
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try some random orderings
    rng = random.Random(42)
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        # Create a shuffled candidate list
        cand = list(range(1, 1001))
        rng.shuffle(cand)
        # But bias towards smaller numbers for better density
        # Actually, let's try a mix: some structured approaches
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    if time.time() < deadline - 5.0:
        ls_time = min(10.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            best = _local_search(best, time_limit=ls_time)
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        # Fallback to standard greedy result
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
