import time
import random
import numpy as np

N = 16
DEADLINE = 300.0  # seconds, leave margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 15.0  # 15s safety margin
    
    points = [(x, y) for x in range(N) for y in range(N)]
    npts = len(points)
    
    # Precompute isosceles conflicts using bitmasks for speed
    # pair_conflict[i][j] = bitmask of k such that i, j, k form an isosceles triangle
    # Using a list of lists of Python integers for fast bitwise ops
    pair_conflict = [[0]*npts for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            mask = 0
            for k in range(npts):
                if k != i and k != j and _is_isosceles(points[i], points[j], points[k]):
                    mask |= (1 << k)
            pair_conflict[i][j] = mask
            pair_conflict[j][i] = mask
    
    # Helper: compute blocked mask for a given set S
    def compute_blocked(S):
        blocked = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                blocked |= pair_conflict[S_list[i]][S_list[j]]
        return blocked
    
    # Helper: check if S is maximal
    def is_maximal(S, blocked=None):
        if blocked is None:
            blocked = compute_blocked(S)
        for p in range(npts):
            if p not in S and not (blocked & (1 << p)):
                return False
        return True
    
    # Helper: validate that S has no isosceles triples
    def is_valid(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    if _is_isosceles(points[S_list[i]], points[S_list[j]], points[S_list[k]]):
                        return False
        return True

    # Optimized Greedy construction
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = set()
        S_list = []
        blocked = 0
        
        while True:
            if time.time() > deadline:
                break
            
            # Find available points
            avail_list = []
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    avail_list.append(p)
            
            if not avail_list:
                break
            
            # Score each available point
            best_p = None
            best_score = -1
            rng.shuffle(avail_list)
            
            for p in avail_list:
                new_blocked = 0
                for a in S:
                    new_blocked |= pair_conflict[p][a]
                new_blocked &= ~blocked
                new_blocked &= ~(1 << p)
                score = bin(new_blocked).count('1')
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            S.add(best_p)
            S_list.append(best_p)
            for a in S:
                if a != best_p:
                    blocked |= pair_conflict[best_p][a]
        
        # Ensure maximality by adding any remaining valid points
        changed = True
        while changed:
            changed = False
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    S.add(p)
                    S_list.append(p)
                    for a in S:
                        if a != p:
                            blocked |= pair_conflict[p][a]
                    changed = True
        
        return S_list, blocked

    # Optimized Local Search: 1-1 swap
    def local_search(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        blocked = compute_blocked(S)
        
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= best_size:
                continue
            
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            for remove_p in S_list:
                if len(S) - 1 >= best_size:
                    continue
                
                S.remove(remove_p)
                new_blocked = compute_blocked(S)
                
                # Check if S - {p} is maximal
                is_max = is_maximal(S, new_blocked)
                if is_max:
                    best_S = list(S)
                    best_size = len(S)
                    improved = True
                    blocked = new_blocked
                    break
                
                S.add(remove_p)
            
            if not improved:
                break
        
        return best_S

    # New: 2-point removal local search
    def local_search_2remove(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        blocked = compute_blocked(S)
        
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= best_size:
                continue
            
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            # Try removing pairs of points
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if len(S) - 2 >= best_size:
                        continue
                    
                    p1 = S_list[i]
                    p2 = S_list[j]
                    
                    S.remove(p1)
                    S.remove(p2)
                    new_blocked = compute_blocked(S)
                    
                    # Check if S - {p1, p2} is maximal
                    is_max = is_maximal(S, new_blocked)
                    if is_max:
                        best_S = list(S)
                        best_size = len(S)
                        improved = True
                        blocked = new_blocked
                        break
                    
                    S.add(p1)
                    S.add(p2)
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_S

    # Main
    best_result = None
    best_size = npts
    
    # Baseline
    baseline_S, _ = greedy_construct(42)
    if baseline_S:
        best_result = baseline_S
        best_size = len(baseline_S)
    
    # Multiple trials
    num_trials = 100
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        seed = 1000 + trial
        S, _ = greedy_construct(seed)
        if not S:
            continue
        
        # Apply 1-point removal local search
        improved_S = local_search(S, seed)
        
        # Apply 2-point removal local search
        improved_S_2 = local_search_2remove(improved_S, seed + 10000)
        
        if len(improved_S_2) < best_size:
            best_size = len(improved_S_2)
            best_result = improved_S_2
    
    # Final validation
    if best_result is not None:
        S_set = set(best_result)
        if not is_valid(S_set) or not is_maximal(S_set):
            if baseline_S and is_valid(set(baseline_S)) and is_maximal(set(baseline_S)):
                best_result = baseline_S
            else:
                best_result = []
    
    return [points[i] for i in best_result] if best_result else [points[0]]
