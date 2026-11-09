import time
import random
import numpy as np

DEADLINE = time.time() + 120.0
N = 1000

def _sum_counts(indicator):
    # number of pairs (a<=b) in the set that sum to each s
    return np.convolve(indicator, indicator)

def _valid(indicator):
    return bool(np.all(_sum_counts(indicator) <= 2))

def _greedy_random(seed=None):
    rng = random.Random(seed)
    counts = np.zeros(2 * N + 1, dtype=np.int64)
    chosen = []
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok and counts[2 * x] + 1 > 2:
            ok = False
        if ok:
            chosen.append(x)
            for y in chosen:
                counts[x + y] += 1
    return chosen

def _greedy_sequential():
    counts = np.zeros(2 * N + 1, dtype=np.int64)
    chosen = []
    for x in range(1, N + 1):
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok and counts[2 * x] + 1 > 2:
            ok = False
        if ok:
            chosen.append(x)
            for y in chosen:
                counts[x + y] += 1
    return chosen

def _local_search(chosen, time_budget=1.0):
    """Local search with add, remove, and swap moves."""
    ind = np.zeros(N + 1, dtype=np.int8)
    for x in chosen:
        ind[x] = 1
    cur = set(chosen)
    t_start = time.time()
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - t_start) < time_budget:
        improved = False
        no_improve_count += 1
        if no_improve_count > 50:
            break
            
        # Try swap moves: remove x, add y
        for x in list(cur):
            if (time.time() - t_start) >= time_budget:
                break
            ind[x] = 0
            base = _sum_counts(ind)
            # Find best y to add
            best_y = None
            best_gain = 0
            for y in range(1, N + 1):
                if y in cur:
                    continue
                # Check if adding y is valid
                if base[2 * y] >= 2:
                    continue
                # Check all sums x+y for x in cur
                valid = True
                for z in cur:
                    if base[z + y] >= 2:
                        valid = False
                        break
                if valid:
                    # Gain: we removed x and added y, size stays same
                    # But we might enable future additions
                    if best_y is None:
                        best_y = y
                        best_gain = 0
                    break
            
            if best_y is not None and best_y not in cur:
                cur.remove(x)
                cur.add(best_y)
                ind[x] = 0
                ind[best_y] = 1
                improved = True
                no_improve_count = 0
                continue
            else:
                ind[x] = 1
        
        # Try add moves: add a new element
        ind_full = _sum_counts(ind)
        added = False
        for y in range(1, N + 1):
            if (time.time() - t_start) >= time_budget:
                break
            if y in cur:
                continue
            if ind_full[2 * y] >= 2:
                continue
            valid = True
            for z in cur:
                if ind_full[z + y] >= 2:
                    valid = False
                    break
            if valid:
                cur.add(y)
                ind[y] = 1
                improved = True
                added = True
                no_improve_count = 0
                break
        
        # Try remove moves if stuck
        if not improved and no_improve_count > 10:
            # Remove a random element to try different configuration
            if cur:
                x = random.choice(list(cur))
                cur.remove(x)
                ind[x] = 0
                improved = True
                no_improve_count = 0
    
    return cur

def _construct_single(seed=None):
    """Run one construction attempt with local search."""
    t_start = time.time()
    
    # Choose initial greedy strategy
    if seed is None:
        chosen = _greedy_sequential()
    else:
        chosen = _greedy_random(seed)
    
    # Apply local search with time budget
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 0.5:
        time_budget = min(remaining, 2.0)
        chosen = _local_search(chosen, time_budget)
    
    return set(chosen)

def construct():
    t0 = time.time()
    best = set()
    best_size = 0
    
    # Multiple restarts with different seeds
    attempts = 0
    while time.time() < DEADLINE - 3.0:
        attempts += 1
        seed = attempts * 12345 + 67890
        cand = _construct_single(seed)
        
        if len(cand) > best_size:
            best_size = len(cand)
            best = cand
            # If we found a good solution, do more local search on it
            remaining = DEADLINE - time.time() - 2.0
            if remaining > 1.0:
                best = _local_search(best, min(remaining, 3.0))
        
        # Early termination if we've been stuck for a while
        if attempts > 20 and len(cand) <= best_size - 5:
            # Try a few more but with less time each
            pass
        
        # Limit total attempts to avoid too many tiny ones
        if attempts >= 50:
            break
    
    # Final local search on best if time remains
    remaining = DEADLINE - time.time() - 1.0
    if remaining > 0.5:
        best = _local_search(best, min(remaining, 2.0))
    
    return sorted(best)
