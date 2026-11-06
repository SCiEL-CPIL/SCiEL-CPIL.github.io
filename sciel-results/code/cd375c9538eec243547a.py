"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _greedy_construct():
    """Greedy construction of a B2[2] set in [1, 1000]."""
    chosen = []
    counts = defaultdict(int)
    for x in range(1, 1001):
        # Check if adding x keeps all sums at most 2
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                ok = False
                break
            new_sums.append(s)
        if ok:
            s2 = 2 * x
            if counts[s2] >= 2:
                ok = False
            else:
                new_sums.append(s2)
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
    return chosen

def _validate(chosen):
    """Verify that every sum has at most 2 representations."""
    if len(chosen) != len(set(chosen)):
        return False
    if any(x < 1 or x > 1000 for x in chosen):
        return False
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            s = chosen[i] + chosen[j]
            counts[s] += 1
            if counts[s] > 2:
                return False
    return True

def _try_local_search(initial, time_limit=100):
    """Try to improve the set via local search (swap/add operations)."""
    chosen = list(initial)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    def add_element(x):
        nonlocal counts, n
        new_sums = []
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                ok = False
                break
            new_sums.append(s)
        if ok:
            s2 = 2 * x
            if counts[s2] >= 2:
                ok = False
            else:
                new_sums.append(s2)
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            n += 1
            return True
        return False
    
    def remove_element(x):
        nonlocal counts, n
        idx = chosen.index(x)
        # Remove sums involving x
        for j in range(len(chosen)):
            if j == idx:
                continue
            s = x + chosen[j]
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s2 = 2 * x
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]
        chosen.pop(idx)
        n -= 1
    
    start_time = time.time()
    best_size = len(chosen)
    
    # Try to add elements
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        # Find a random unused element
        used = set(chosen)
        candidates = [x for x in range(1, 1001) if x not in used]
        if not candidates:
            break
        random.shuffle(candidates)
        added = False
        for x in candidates:
            if add_element(x):
                best_size = max(best_size, len(chosen))
                added = True
                break
        if not added:
            # Try removing a random element and then adding
            if len(chosen) > 10:
                idx = random.randint(0, len(chosen) - 1)
                x_remove = chosen[idx]
                remove_element(x_remove)
                # Try to add two new elements
                added_count = 0
                for x in candidates:
                    if time.time() - start_time > time_limit:
                        break
                    if add_element(x):
                        added_count += 1
                        if added_count >= 2:
                            break
                if added_count < 1:
                    # Revert
                    add_element(x_remove)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with greedy construction
    best = _greedy_construct()
    
    # Try local search improvements
    random.seed(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = _try_local_search(best, time_limit=min(20, deadline - time.time()))
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _validate(best):
        # Fallback to pure greedy
        best = _greedy_construct()
    
    return best
# EVOLVE-BLOCK-END
