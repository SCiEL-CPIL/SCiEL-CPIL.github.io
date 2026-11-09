import time
import random

def _find_conflict(S):
    """
    Finds a subset of S that sums to 200.
    Returns the subset (list) or None.
    S is a list of distinct integers.
    """
    target = 200
    reach = {0: []}
    
    for x in S:
        new_reach = {}
        for s, subset in reach.items():
            if s + x <= target:
                if s + x not in new_reach:
                    new_reach[s + x] = subset + [x]
        
        for s, subset in new_reach.items():
            if s not in reach or len(subset) < len(reach[s]):
                reach[s] = subset
    
    if target in reach:
        return reach[target]
    return None

def _greedy_low_degree():
    """
    Greedy approach: build a conflict graph where edges connect pairs
    that together with some other elements could sum to 200.
    Use a simpler heuristic: iteratively remove the element that
    participates in the most minimal conflicts.
    """
    S = list(range(1, 101))
    
    while True:
        conflict = _find_conflict(S)
        if conflict is None:
            return S
        
        # Count how many times each element appears in conflicts
        # by finding multiple conflicts
        conflict_counts = {x: 0 for x in S}
        for _ in range(10):
            c = _find_conflict(S)
            if c is None:
                break
            for x in c:
                conflict_counts[x] += 1
        
        # Remove the element with highest conflict count
        max_count = max(conflict_counts.values())
        candidates = [x for x in S if conflict_counts[x] == max_count]
        elem_to_remove = max(candidates)  # Tie-break: remove largest
        S.remove(elem_to_remove)
    
    return S

def _greedy_greedy_addition():
    """
    Start with empty set, add elements one by one if they don't create a conflict.
    Try multiple orderings.
    """
    best = []
    for seed in range(20):
        random.seed(seed)
        S = list(range(1, 101))
        random.shuffle(S)
        
        selected = []
        for x in S:
            test_set = selected + [x]
            if _find_conflict(test_set) is None:
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Deterministic descending greedy
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    if not best_set or len(selected) > best_size:
        best_set = selected
        best_size = len(selected)
    
    # Strategy 2: Greedy low-degree removal
    if time.time() - start_time < deadline - 10:
        result = _greedy_low_degree()
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 3: Greedy addition with multiple random orders
    if time.time() - start_time < deadline - 5:
        result = _greedy_greedy_addition()
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 4: Randomized local search with improved removal heuristic
    while time.time() - start_time < deadline - 2:
        S = list(range(1, 101))
        random.shuffle(S)
        
        while True:
            conflict = _find_conflict(S)
            if conflict is None:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                break
            
            # Try multiple removal strategies
            if len(conflict) > 1:
                # Remove the element that appears in most conflicts (sampled)
                conflict_counts = {x: 0 for x in conflict}
                for _ in range(5):
                    c = _find_conflict(S)
                    if c is None:
                        break
                    for x in c:
                        if x in conflict_counts:
                            conflict_counts[x] += 1
                max_count = max(conflict_counts.values())
                candidates = [x for x in conflict if conflict_counts[x] == max_count]
                elem_to_remove = random.choice(candidates)
            else:
                elem_to_remove = conflict[0]
            
            S.remove(elem_to_remove)
            
            if time.time() - start_time > deadline - 2.0:
                break
    
    return best_set
