import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True

def _get_conflicts(x, S):
    """
    Return the set of elements in S that conflict with x.
    If x is added, these elements (or others) might need to be removed.
    Actually, we just need to know if x is addable.
    This function is for analysis, not directly used in _can_add.
    """
    conflicts = set()
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                conflicts.add(b)
                conflicts.add(c)
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                conflicts.add(a)
                conflicts.add(c)
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                conflicts.add(a)
                conflicts.add(b)
                
    return conflicts

def _local_search(S, deadline):
    """
    Local search to maximize size of S.
    """
    best_set = S.copy()
    best_size = len(S)
    current_set = S.copy()
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        candidates = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if _can_add(x, current_set):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                break
        
        if added:
            continue
        
        # 2. Try 1-1 swaps
        swapped = False
        in_set = list(current_set)
        out_set = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(in_set)
        random.shuffle(out_set)
        
        for y in in_set:
            for x in out_set:
                S_temp = current_set - {y}
                if _can_add(x, S_temp):
                    current_set = S_temp
                    current_set.add(x)
                    swapped = True
                    # Check if this swap enables adds (handled in next loop)
                    # But we can check if size increased? No, size is same.
                    # We just want to change the set to maybe allow adds later.
                    break
            if swapped:
                break
        
        if swapped:
            continue
            
        # 3. Try 1-2 swaps (Remove 1, Add 2)
        # This is more expensive but can increase size.
        improved = False
        in_set = list(current_set)
        out_set = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(in_set)
        
        for y in in_set:
            if time.time() > deadline - 5:
                break
            S_temp = current_set - {y}
            # Try to add 2 elements to S_temp
            # We need to find x1, x2 such that S_temp + {x1, x2} is valid.
            # This is equivalent to:
            # 1. x1 addable to S_temp
            # 2. x2 addable to S_temp + {x1}
            
            x1_candidates = [x for x in out_set if x != y]
            random.shuffle(x1_candidates)
            
            for x1 in x1_candidates:
                if time.time() > deadline - 5:
                    break
                if _can_add(x1, S_temp):
                    S_temp1 = S_temp | {x1}
                    x2_candidates = [x for x in out_set if x != y and x != x1]
                    random.shuffle(x2_candidates)
                    for x2 in x2_candidates:
                        if _can_add(x2, S_temp1):
                            current_set = S_temp | {x1, x2}
                            improved = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set.copy()
                            break
                    if improved:
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # 4. If stuck, try a random restart or revert to best
        # Revert to best and try again with different randomization
        current_set = best_set.copy()
        # To avoid infinite loop, we could break, but we have time.
        # Let's just break out of the inner loop and let the outer while handle it?
        # No, the outer while is the main loop.
        # If we can't improve, we are at a local optimum.
        # We should break to save time for other strategies?
        # Or just return best_set.
        # Given the time budget, if we are stuck, further random swaps are unlikely to help
        # without a more complex move (like 2-2).
        # We will implement a simple 2-2 swap if stuck for a while.
        
        # For now, break if we can't find any 1-2 improvement.
        # This is a heuristic.
        break
        
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Generate initial candidates
    candidates = []
    
    # 1. Intervals [L, 256]
    for L in range(150, 257):
        S = set(range(L, 257))
        if _validate(S):
            candidates.append(S)
            
    # 2. Greedy descending
    S_greedy = set()
    for x in range(N, 0, -1):
        if _can_add(x, S_greedy):
            S_greedy.add(x)
    candidates.append(S_greedy)
    
    # 3. Greedy ascending
    S_greedy_asc = set()
    for x in range(1, N + 1):
        if _can_add(x, S_greedy_asc):
            S_greedy_asc.add(x)
    candidates.append(S_greedy_asc)
    
    # 4. Random restarts
    for i in range(10):
        random.seed(i)
        S_rand = set()
        candidates_list = list(range(1, N + 1))
        random.shuffle(candidates_list)
        for x in candidates_list:
            if _can_add(x, S_rand):
                S_rand.add(x)
        candidates.append(S_rand)
        
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    if not best_set:
        best_set = {N}
        best_size = 1
        
    # Run Local Search on the best candidate
    if time.time() < deadline - 5:
        best_set = _local_search(best_set, deadline - 5)
        
    # If time remains, try local search on other good candidates
    if time.time() < deadline - 5:
        # Find other candidates with size close to best
        other_candidates = [S for S in candidates if _validate(S) and len(S) >= best_size - 5 and S != best_set]
        random.shuffle(other_candidates)
        
        for S in other_candidates:
            if time.time() > deadline - 5:
                break
            candidate_best = _local_search(S, deadline - 5)
            if len(candidate_best) > len(best_set):
                best_set = candidate_best
                
    # Final polish: try simple additions
    while time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if _can_add(x, best_set):
                best_set.add(x)
                changed = True
        if not changed:
            break
            
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
