"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _triple_sums_involving(x, S):
    """Compute all unordered triple sums that include x, given sorted list S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _greedy_one_pass(order, deadline):
    """Greedy construction with a given candidate order. Returns (selected, sums_set, sel_set)."""
    selected = []
    existing_sums_set = set()
    selected_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        new_sums = _triple_sums_involving(x, selected)
        collision = False
        for s in new_sums:
            if s in existing_sums_set:
                collision = True
                break
        if not collision:
            selected.append(x)
            selected_set.add(x)
            for s in new_sums:
                existing_sums_set.add(s)
    return selected, existing_sums_set, selected_set

def _local_search_swap(selected, existing_sums_set, selected_set, all_candidates, deadline, max_attempts=40):
    """Try swap-based local search to improve the set."""
    n = len(selected)
    if n == 0:
        return selected, existing_sums_set, selected_set
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        idx = random.randrange(n)
        removed = selected[idx]
        # Find candidates not in selected_set
        not_in_set = [c for c in all_candidates if c not in selected_set]
        if not not_in_set:
            break
        add = random.choice(not_in_set)
        if add == removed:
            continue
        # Build new set and check validity
        new_selected = [s for i, s in enumerate(selected) if i != idx]
        new_selected.append(add)
        new_selected.sort()
        # Check all triple sums unique
        all_sums = set()
        valid = True
        for t in combinations_with_replacement(new_selected, 3):
            s = sum(t)
            if s in all_sums:
                valid = False
                break
            all_sums.add(s)
        if valid:
            selected = new_selected
            selected_set.discard(removed)
            selected_set.add(add)
            existing_sums_set = all_sums
    return selected, existing_sums_set, selected_set

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n_max = 4096
    all_candidates = list(range(1, n_max + 1))
    
    best_selected = []
    best_size = 0
    
    # Deterministic greedy first
    selected, sums_set, sel_set = _greedy_one_pass(list(range(1, n_max + 1)), deadline)
    best_selected = selected[:]
    best_size = len(selected)
    
    # Randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 2:
        order = all_candidates[:]
        rng.shuffle(order)
        selected, sums_set, sel_set = _greedy_one_pass(order, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
        
        # Local search on the best so far if we have time
        if time.time() < deadline - 5:
            selected, sums_set, sel_set = _local_search_swap(
                selected, sums_set, sel_set, all_candidates, deadline, max_attempts=25
            )
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected[:]
    
    return best_selected
