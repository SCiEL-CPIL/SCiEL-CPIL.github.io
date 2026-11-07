"""Rewritten program: maximize size of a set of distinct ints in 1..256 such that
no two distinct non-empty subset sums divide one another (and no two subsets have the same sum)."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_subset_sums(elements):
    """Return the set of all non-empty subset sums of elements."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check if the set of elements satisfies the divisibility-incomparability condition."""
    sums = _all_subset_sums(elements)
    sums_list = sorted(sums)
    for i, a in enumerate(sums_list):
        for b in sums_list[i+1:]:
            if b % a == 0:
                return False
    return True

def _greedy_construct(seed=0):
    """Greedy construction: start with large numbers, add smaller ones if valid."""
    rng = random.Random(seed)
    # Start with a pool of candidate numbers
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    result = []
    for c in candidates:
        if len(result) >= 30:
            break
        trial = result + [c]
        if _is_valid(trial):
            result.append(c)
    
    return result

def _local_search(elements, seed=0, time_limit=30):
    """Local search: try to add elements from 1..256 not in the set."""
    rng = random.Random(seed)
    elements = list(elements)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if len(elements) >= 30:
            break
        # Try adding a random element not in the set
        candidates = [x for x in range(1, 257) if x not in elements]
        if not candidates:
            break
        rng.shuffle(candidates)
        improved = False
        for c in candidates:
            trial = elements + [c]
            if _is_valid(trial):
                elements = trial
                improved = True
                break
        if not improved:
            # Try removing one element and adding two others
            if len(elements) >= 2:
                for i in range(len(elements)):
                    for j in range(len(elements)):
                        if i == j:
                            continue
                        trial = elements[:i] + elements[i+1:]
                        # Try to add two new elements
                        candidates2 = [x for x in range(1, 257) if x not in trial]
                        rng.shuffle(candidates2)
                        added = 0
                        for c in candidates2:
                            if added >= 2:
                                break
                            trial2 = trial + [c]
                            if _is_valid(trial2):
                                trial = trial2
                                added += 1
                        if len(trial) > len(elements):
                            elements = trial
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            # Try removing one element
            if len(elements) >= 1:
                for i in range(len(elements)):
                    trial = elements[:i] + elements[i+1:]
                    # Try to add one new element
                    candidates3 = [x for x in range(1, 257) if x not in trial]
                    rng.shuffle(candidates3)
                    for c in candidates3:
                        trial2 = trial + [c]
                        if _is_valid(trial2):
                            elements = trial2
                            break
            if not improved:
                break
    
    return elements

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s limit
    
    best = []
    
    # Try multiple greedy starts with different seeds
    for seed in range(50):
        if time.time() - start_time > deadline:
            break
        result = _greedy_construct(seed)
        if len(result) > len(best):
            best = result
        if len(best) >= 30:
            break
    
    # Local search on the best result
    if time.time() - start_time < deadline:
        improved = _local_search(best, seed=42, time_limit=min(30, deadline - (time.time() - start_time)))
        if len(improved) > len(best):
            best = improved
    
    # Final local search attempts with different seeds if time permits
    if time.time() - start_time < deadline:
        for seed in [7, 13, 99, 123, 555]:
            if time.time() - start_time > deadline:
                break
            improved = _local_search(best, seed=seed, time_limit=min(10, deadline - (time.time() - start_time)))
            if len(improved) > len(best):
                best = improved
    
    # Ensure we return a valid list of Python ints
    best = [int(x) for x in best]
    return best

# EVOLVE-BLOCK-END
