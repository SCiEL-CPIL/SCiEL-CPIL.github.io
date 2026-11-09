import time
import random

def _find_conflict_fast(S):
    """
    Fast subset-sum check using boolean array.
    Returns True if any subset of S sums to 200, else False.
    """
    target = 200
    reachable = [False] * (target + 1)
    reachable[0] = True
    
    for x in S:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if reachable[s]:
                reachable[s + x] = True
                if s + x == target:
                    return True
    
    return reachable[target]

def _find_conflict_with_witness(S):
    """
    Find a subset of S summing to 200.
    Returns the subset (list) or None.
    """
    target = 200
    reachable = [False] * (target + 1)
    reachable[0] = True
    parent = [None] * (target + 1)
    
    for x in S:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if reachable[s] and not reachable[s + x]:
                reachable[s + x] = True
                parent[s + x] = (s, x)
                if s + x == target:
                    subset = []
                    cur = target
                    while cur > 0:
                        prev_s, elem = parent[cur]
                        subset.append(elem)
                        cur = prev_s
                    return subset
    
    if reachable[target]:
        subset = []
        cur = target
        while cur > 0:
            prev_s, elem = parent[cur]
            subset.append(elem)
            cur = prev_s
        return subset
    
    return None

def _greedy_low_degree():
    """
    Greedy removal: iteratively remove an element from a found conflict.
    Use heuristic to pick which element to remove.
    """
    S = list(range(1, 101))
    
    while True:
        if not _find_conflict_fast(S):
            return S
        
        conflict = _find_conflict_with_witness(S)
        if conflict is None:
            return S
        
        # Heuristic: remove the largest element from the conflict
        # This tends to reduce the sum space more
        elem_to_remove = max(conflict)
        S.remove(elem_to_remove)
    
    return S

def _greedy_addition():
    """
    Start with empty set, add elements one by one if they don't create a conflict.
    Try multiple orderings.
    """
    best = []
    for seed in range(50):
        random.seed(seed)
        S = list(range(1, 101))
        random.shuffle(S)
        
        selected = []
        for x in S:
            test_set = selected + [x]
            if not _find_conflict_fast(test_set):
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
    
    return best

def _local_search(S):
    """
    Local search: try to add elements that don't create conflicts.
    """
    S_set = set(S)
    improved = True
    while improved:
        improved = False
        for x in range(1, 101):
            if x in S_set:
                continue
            test_set = list(S) + [x]
            if not _find_conflict_fast(test_set):
                S.append(x)
                S_set.add(x)
                improved = True
    
    return S

def _greedy_descending():
    """
    Deterministic descending greedy.
    """
    selected = []
    reachable = [False] * 201
    reachable[0] = True
    
    for x in range(100, 0, -1):
        new = [False] * 201
        for s in range(200 - x, -1, -1):
            if reachable[s]:
                new[s + x] = True
        if new[200]:
            continue
        for s in range(201):
            if new[s]:
                reachable[s] = True
        selected.append(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = 80.0
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Deterministic descending greedy
    selected = _greedy_descending()
    if not _find_conflict_fast(selected):
        best_set = selected
        best_size = len(selected)
    
    # Strategy 2: Greedy low-degree removal
    if time.time() - start_time < deadline - 20:
        result = _greedy_low_degree()
        if not _find_conflict_fast(result) and len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 3: Greedy addition with multiple random orders
    if time.time() - start_time < deadline - 10:
        result = _greedy_addition()
        if not _find_conflict_fast(result) and len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 4: Local search on best set
    if best_set is not None and time.time() - start_time < deadline - 5:
        S = best_set[:]
        S = _local_search(S)
        if not _find_conflict_fast(S) and len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy 5: Randomized restarts
    while time.time() - start_time < deadline - 2:
        S = list(range(1, 101))
        random.shuffle(S)
        
        while True:
            if not _find_conflict_fast(S):
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                break
            
            conflict = _find_conflict_with_witness(S)
            if conflict is None:
                break
            
            # Remove a random element from the conflict
            elem_to_remove = random.choice(conflict)
            S.remove(elem_to_remove)
        
        if time.time() - start_time > deadline - 2.0:
            break
    
    if best_set is None:
        best_set = [1, 2, 3, 4, 5]
    
    return best_set
