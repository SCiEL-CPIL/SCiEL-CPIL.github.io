"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if selected set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    if n < 3:
        return True
    # Check if x divides sum of any 3 existing elements
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
    # Check if any existing element divides sum of x + any 2 other existing elements
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        # Precompute pair sums for efficiency
        for p0_idx in range(len(others)):
            v0 = others[p0_idx]
            for p1_idx in range(p0_idx + 1, len(others)):
                s = x + v0 + others[p1_idx]
                if s % a == 0:
                    return False
    return True

def _greedy(order):
    """Greedy construction using given order of candidates."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve selected set via swaps and additions."""
    best = list(selected)
    best_size = len(best)
    all_nums = set(range(1, 193))
    missing = sorted(all_nums - set(best), reverse=True)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding any missing number
        for x in missing:
            if _can_add(best, x):
                best.append(x)
                missing.remove(x)
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        found_swap = False
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            removed = best[i]
            for x in missing:
                if x == removed:
                    continue
                trial = best[:i] + best[i+1:] + [x]
                if _check_valid(trial):
                    best = trial
                    best_size = len(best)
                    missing.remove(x)
                    missing.append(removed)
                    missing.sort(reverse=True)
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps: remove two, add one (net -1, but may enable future adds)
        # Actually, try 1-for-2: remove one, add two (net +1)
        found_1for2 = False
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            removed = best[i]
            base = best[:i] + best[i+1:]
            for x1 in missing:
                if x1 == removed:
                    continue
                if not _can_add(base, x1):
                    continue
                trial1 = base + [x1]
                for x2 in missing:
                    if x2 == removed or x2 == x1:
                        continue
                    if _can_add(trial1, x2):
                        trial = trial1 + [x2]
                        if _check_valid(trial):
                            best = trial
                            best_size = len(best)
                            missing.remove(x1)
                            missing.remove(x2)
                            missing.append(removed)
                            missing.sort(reverse=True)
                            improved = True
                            found_1for2 = True
                            break
                if found_1for2:
                    break
            if found_1for2:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Try multiple greedy orderings
    candidates = list(range(1, 193))
    orders = [
        list(range(192, 0, -1)),  # Descending
        list(range(1, 193)),      # Ascending
        list(range(192, 0, -2)),  # Even descending
        list(range(191, 0, -2)),  # Odd descending
    ]
    
    # Add randomized orders with different seeds
    rng = random.Random(42)
    for _ in range(15):
        if time.time() > deadline - 40:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    best_result = []
    for order in orders:
        if time.time() > deadline - 30:
            break
        result = _greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search improvement
    if time.time() < deadline - 5:
        improved = _local_search(best_result, deadline)
        if len(improved) > len(best_result):
            best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
