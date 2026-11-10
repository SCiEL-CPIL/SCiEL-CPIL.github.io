"""Rewritten program: construct a subset of [1,256] with no 4-term AP, maximizing 3-term APs."""
import time
import random
import numpy as np

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def _has_4ap(arr_set):
    """Check if a set has a non-constant 4-term AP using numpy for speed."""
    arr = np.array(sorted(arr_set))
    n = len(arr)
    if n < 4:
        return False
    # Use set for O(1) lookups
    S = set(arr)
    for i in range(n):
        a = arr[i]
        max_d = (N - a) // 3
        if max_d < 1:
            continue
        # Check all d
        for d in range(1, max_d + 1):
            if (a + d in S and a + 2*d in S and a + 3*d in S):
                return True
    return False

def _count_3ap(arr_set):
    """Count distinct 3-term APs (positive difference) in the set."""
    S = set(arr_set)
    count = 0
    arr = sorted(arr_set)
    for a in arr:
        max_d = (N - a) // 2
        if max_d < 1:
            continue
        for d in range(1, max_d + 1):
            if a + d in S and a + 2*d in S:
                count += 1
    return count

def _fast_count_3ap(arr_set):
    """Faster count using sorted array and two-pointer or direct iteration."""
    S = set(arr_set)
    count = 0
    arr = sorted(arr_set)
    for a in arr:
        # a is the first term
        max_d = (N - a) // 2
        if max_d < 1:
            continue
        for d in range(1, max_d + 1):
            if a + d in S and a + 2*d in S:
                count += 1
    return count

def _construct_greedy_base():
    """Greedy construction: add elements in order if no 4-AP is formed."""
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # x could be the 4th term: x-3d, x-2d, x-d, x
        for d in range(1, (x) // 3 + 1):
            if x - d in selected and x - 2*d in selected and x - 3*d in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x could be the 3rd term: x-2d, x-d, x, x+d
        for d in range(1, (N - x) // 1 + 1):
            if x - d in selected and x - 2*d in selected and x + d in N and x + d <= N:
                # x+d is not in selected yet, so this doesn't create a 4-AP with x as 3rd
                # Actually, for x to be 3rd term, we need x-2d, x-d, x, x+d all in set
                # x+d is not in selected, so no
                pass
        # x could be the 2nd term: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, N - x) // 1 + 1):
            if x - d in selected and x + d in selected and x + 2*d in selected:
                creates_4ap = True
                break
            if creates_4ap:
                break
        if creates_4ap:
            continue
        # x could be the 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if x + d in selected and x + 2*d in selected and x + 3*d in selected:
                creates_4ap = True
                break
        if not creates_4ap:
            selected.add(x)
    return selected

def _local_search(arr_set, time_budget):
    """Improve the set via local search: try adding/removing elements."""
    start_time = time.time()
    current = set(arr_set)
    current_score = _count_3ap(current)
    
    # Precompute which elements can be added without creating 4-AP
    # Try to add elements
    candidates = [x for x in range(1, N + 1) if x not in current]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        for x in candidates:
            if time.time() - start_time > time_budget:
                break
            new_set = current | {x}
            if not _has_4ap(new_set):
                new_score = _count_3ap(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    candidates.remove(x)
                    improved = True
    
    # Try removing elements that are in 4-APs (shouldn't exist) or that don't contribute much
    # Actually, since we have no 4-APs, try removing elements and see if we can add more
    # Try swaps: remove one, add another
    arr_list = sorted(current)
    for i in range(len(arr_list)):
        if time.time() - start_time > time_budget:
            break
        for x in candidates:
            if time.time() - start_time > time_budget:
                break
            new_set = (current - {arr_list[i]}) | {x}
            if not _has_4ap(new_set):
                new_score = _count_3ap(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    if x in candidates:
                        candidates.remove(x)
                    arr_list = sorted(current)
                    improved = True
                    break
    
    return current, current_score

def construct():
    start_time = time.time()
    
    # Start with greedy base
    base = _construct_greedy_base()
    base_score = _count_3ap(base)
    
    best = base
    best_score = base_score
    
    # Try multiple random restarts with different orderings
    deadline = start_time + TIME_LIMIT
    
    # Try a few different greedy orderings
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        # Shuffle the order of consideration
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        selected = set()
        for x in order:
            if time.time() > deadline - 10:
                break
            # Check if adding x creates a 4-AP
            creates_4ap = False
            # x as 4th term
            for d in range(1, x // 3 + 1):
                if x - d in selected and x - 2*d in selected and x - 3*d in selected:
                    creates_4ap = True
                    break
            if not creates_4ap:
                # x as 3rd term
                for d in range(1, min(x - 1, N - x) + 1):
                    if x - d in selected and x - 2*d in selected and x + d <= N and x + d in selected:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # x as 2nd term
                for d in range(1, min(x - 1, N - x) + 1):
                    if x - d in selected and x + d in selected and x + 2*d in selected:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # x as 1st term
                for d in range(1, (N - x) // 3 + 1):
                    if x + d in selected and x + 2*d in selected and x + 3*d in selected:
                        creates_4ap = True
                        break
            if not creates_4ap:
                selected.add(x)
        
        score = _count_3ap(selected)
        if score > best_score:
            best = selected
            best_score = score
    
    # Local search on the best solution
    if time.time() < deadline - 5:
        time_budget = min(deadline - 5 - time.time(), 15)
        if time_budget > 1:
            improved_set, improved_score = _local_search(best, time_budget)
            if improved_score > best_score:
                best = improved_set
                best_score = improved_score
    
    return sorted(best)
