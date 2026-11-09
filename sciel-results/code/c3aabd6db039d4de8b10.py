"""Rewritten program: search for a large set of distinct ints in 1..256 with
the property that no two distinct non-empty subset sums divide one another,
and no two distinct subsets have the same sum (all 2^n-1 subset sums are distinct).
"""
import time
import itertools
import random
from functools import lru_cache

DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock limit

def _all_subset_sums(lst):
    """Return the set of all non-empty subset sums of lst."""
    n = len(lst)
    sums = set()
    # Use bitmasks
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += lst[i]
        sums.add(s)
    return sums

def _is_valid(lst):
    """Check if lst satisfies the task constraints.
    1. All elements distinct, in 1..256.
    2. All non-empty subset sums are distinct (no two subsets have same sum).
    3. No two distinct subset sums divide one another.
    """
    if len(lst) != len(set(lst)):
        return False
    if any(x < 1 or x > 256 for x in lst):
        return False
    sums = _all_subset_sums(lst)
    # Check divisibility: for each pair of distinct sums a, b, a must not divide b
    slist = sorted(sums)
    for i, a in enumerate(slist):
        for b in slist[i+1:]:
            if b % a == 0:
                return False
    return True

def _greedy_build(starting_set, candidates, max_size):
    """Greedy: try to add candidates one by one if valid."""
    current = list(starting_set)
    for c in candidates:
        if c in current:
            continue
        trial = current + [c]
        if _is_valid(trial):
            current = trial
            if len(current) >= max_size:
                break
    return current

def _local_search(lst, candidates, max_size, time_budget):
    """Local search: try swaps and additions."""
    current = list(lst)
    best = list(current)
    deadline = time.time() + time_budget
    while time.time() < deadline and len(current) < max_size:
        improved = False
        # Try adding a candidate
        for c in candidates:
            if c in current:
                continue
            trial = current + [c]
            if _is_valid(trial):
                current = trial
                best = list(current)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add one
        for i in range(len(current)):
            for c in candidates:
                if c in current:
                    continue
                trial = [x for j, x in enumerate(current) if j != i] + [c]
                if _is_valid(trial):
                    # Only accept if it allows future growth (heuristic: accept if same size or larger)
                    # For now, accept any valid swap that doesn't reduce size
                    if len(trial) >= len(current):
                        current = trial
                        best = list(current)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    max_size = 30
    
    # Strategy: start with known valid small sets, grow greedily, then local search.
    # A good starting point: use large numbers to avoid small subset sums that divide others.
    # Actually, small numbers create small sums which are more likely to divide others.
    # Let's try starting with a set of numbers that are pairwise coprime and large?
    # Or just start with a random valid set and grow.
    
    best = [31, 32, 34]  # Seed set
    if not _is_valid(best):
        best = [1, 2, 4]  # Fallback, but 1 divides everything, so this is invalid!
        # Actually [1] is valid, [1,2] is invalid because 1 divides 2.
        # Let's find a valid starting set.
        # [100, 150, 200] -> sums: 100,150,200,250,300,350,450. 100 divides 200,300,400(no),500(no). 100|200 -> invalid.
        # We need no subset sum to divide another.
        # [127, 129, 131] -> primes. Sums: 127,129,131, 256,258,260, 387. 
        # 127 divides? 127*2=254, 127*3=381. No.
        # 129 divides? 129*2=258. 258 is a sum! So 129|258 -> invalid.
        # We need to be careful.
        
        # Let's use a quick search for a valid seed of size 3-5.
        random.seed(42)
        found_seed = False
        for _ in range(1000):
            cand = random.sample(range(50, 257), 4)
            if _is_valid(cand):
                best = cand
                found_seed = True
                break
        if not found_seed:
            # Try a structured approach
            for a in range(100, 257, 10):
                for b in range(a+1, 257, 10):
                    for c in range(b+1, 257, 10):
                        if _is_valid([a,b,c]):
                            best = [a,b,c]
                            found_seed = True
                            break
                    if found_seed: break
                if found_seed: break
        if not found_seed:
            # Last resort: just return a valid small set if we can find one
            # [256] is valid.
            best = [256]

    # Now grow
    candidates = list(range(1, 257))
    candidates.sort(reverse=True)  # Try large numbers first? Or small?
    # Large numbers might create large sums that don't divide others as easily.
    # But small numbers create small sums. If a small sum s exists, any multiple of s is forbidden.
    # So maybe we want to avoid small sums? Or ensure no multiples exist.
    
    # Let's try growing from the seed.
    current = _greedy_build(best, candidates, max_size)
    best = current
    
    # Local search
    time_left = deadline - time.time()
    if time_left > 5:
        current = _local_search(best, candidates, max_size, time_left - 5)
        if len(current) > len(best):
            best = current
    
    # Try a few random restarts if time permits
    time_left = deadline - time.time()
    while time_left > 10:
        random.seed(int(time.time() * 1000))
        seed_size = random.randint(2, 5)
        seed = random.sample(range(1, 257), seed_size)
        if _is_valid(seed):
            grown = _greedy_build(seed, candidates, max_size)
            if len(grown) > len(best):
                best = grown
            time_left = deadline - time.time()
            if time_left > 5:
                polished = _local_search(grown, candidates, max_size, min(5, time_left - 2))
                if len(polished) > len(best):
                    best = polished
        time_left = deadline - time.time()
    
    # Ensure return type is list of ints
    return [int(x) for x in best]
