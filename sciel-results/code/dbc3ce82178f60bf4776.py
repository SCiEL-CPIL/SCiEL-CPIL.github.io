"""Rewritten program: greedy + local search for divisibility-incomparable subset sums."""
import time
import random
import math

DEADLINE = 85  # seconds; evaluator kills at 90s, leave margin

def _subset_sums_valid(sums_set, new_elem):
    """Check if adding new_elem keeps all subset sums unique and no two divide each other."""
    # Generate new sums: for each existing sum s, s+new_elem must be new
    # And no existing sum divides any other (including new ones)
    new_sums = set()
    for s in sums_set:
        ns = s + new_elem
        if ns in sums_set or ns in new_sums:
            return None  # duplicate sum
        new_sums.add(ns)
    # new_elem itself
    if new_elem in sums_set or new_elem in new_sums:
        return None
    new_sums.add(new_elem)
    
    # Check divisibility: no two distinct sums divide each other
    all_sums = list(sums_set) + list(new_sums)
    for i in range(len(all_sums)):
        a = all_sums[i]
        for j in range(len(all_sums)):
            if i == j:
                continue
            b = all_sums[j]
            if b > a and b % a == 0:
                return None
            if a > b and a % b == 0:
                return None
    return new_sums

def _try_add(set_elems, sums_set, candidate):
    """Try to add candidate to the set. Returns (new_sums_set, success)."""
    new_sums = _subset_sums_valid(sums_set, candidate)
    if new_sums is None:
        return None, False
    return new_sums, True

def _greedy_build(start_time, deadline, candidates):
    """Greedy: add elements one by one, choosing the best valid candidate."""
    current_set = []
    sums_set = set()
    
    for cand in candidates:
        if time.time() > deadline:
            break
        new_sums, ok = _try_add(current_set, sums_set, cand)
        if ok:
            current_set.append(cand)
            sums_set = sums_set | new_sums
    
    return current_set

def _local_search(current_set, start_time, deadline):
    """Try to improve by swapping or adding elements."""
    if not current_set:
        return current_set
    
    sums_set = set()
    for e in current_set:
        new_sums = {e}
        for s in list(sums_set):
            new_sums.add(s + e)
        sums_set = sums_set | new_sums
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_set)
        
        # Try adding a new element
        for cand in range(1, 257):
            if time.time() > deadline:
                break
            if cand in current_set:
                continue
            new_sums, ok = _try_add(current_set, sums_set, cand)
            if ok:
                current_set.append(cand)
                sums_set = sums_set | new_sums
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved:
            for i in range(n):
                if time.time() > deadline:
                    break
                # Rebuild sums without element i
                temp_set = current_set[:i] + current_set[i+1:]
                temp_sums = set()
                for e in temp_set:
                    new_sums = {e}
                    for s in list(temp_sums):
                        new_sums.add(s + e)
                    temp_sums = temp_sums | new_sums
                
                # Try to add a new element
                for cand in range(1, 257):
                    if time.time() > deadline:
                        break
                    if cand in temp_set:
                        continue
                    new_sums, ok = _try_add(temp_set, temp_sums, cand)
                    if ok:
                        current_set = temp_set + [cand]
                        sums_set = temp_sums | new_sums
                        improved = True
                        break
                if improved:
                    break
    
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_set = []
    
    # Strategy 1: Greedy from largest to smallest (large numbers have fewer divisors)
    candidates_desc = list(range(256, 0, -1))
    s1 = _greedy_build(start_time, deadline, candidates_desc)
    if len(s1) > len(best_set):
        best_set = s1
    
    if time.time() > deadline - 5:
        return best_set
    
    # Strategy 2: Greedy from smallest to largest
    candidates_asc = list(range(1, 257))
    s2 = _greedy_build(start_time, deadline, candidates_asc)
    if len(s2) > len(best_set):
        best_set = s2
    
    if time.time() > deadline - 10:
        return best_set
    
    # Strategy 3: Randomized greedy with multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        random.shuffle(candidates_desc)
        s = _greedy_build(start_time, deadline, candidates_desc)
        if len(s) > len(best_set):
            best_set = s
    
    if time.time() > deadline - 10:
        return best_set
    
    # Strategy 4: Local search from the best found so far
    if best_set:
        improved = _local_search(list(best_set), start_time, deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
