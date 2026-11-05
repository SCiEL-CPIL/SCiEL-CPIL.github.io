import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result

def _build_conflict_graph():
    """
    Build conflict graph where edge (u,v) means u and v cannot both be in a valid set.
    Edge exists if 1/u = sum of reciprocals of some non-empty subset of [1,24] \ {u},
    and v is in that subset. Similarly for v.
    """
    n = 24
    elements = list(range(1, 25))
    conflict = [[False] * n for _ in range(n)]
    
    for idx_u, u in enumerate(elements):
        # Find all non-empty subsets of elements \ {u} whose reciprocals sum to 1/u
        others = [v for v in elements if v != u]
        n_others = len(others)
        
        # Use LCM to work with integers
        lcm_val = _lcm_list(others)
        target_int = lcm_val // u
        values_int = [lcm_val // v for v in others]
        
        # Find all subsets that sum to target_int using meet-in-the-middle
        half = n_others // 2
        left = values_int[:half]
        right = values_int[half:]
        
        left_sums = {}  # sum -> list of masks
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            if s <= target_int:
                if s not in left_sums:
                    left_sums[s] = []
                left_sums[s].append(mask)
        
        right_sums = {}  # sum -> list of masks
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if s <= target_int:
                if s not in right_sums:
                    right_sums[s] = []
                right_sums[s].append(mask)
        
        # Check if target can be formed
        # Right alone
        if target_int in right_sums:
            for rmask in right_sums[target_int]:
                for i in range(len(right)):
                    if rmask & (1 << i):
                        v = others[half + i]
                        v_idx = elements.index(v)
                        conflict[idx_u][v_idx] = True
                        conflict[v_idx][idx_u] = True
        
        # Left alone
        if target_int in left_sums:
            for lmask in left_sums[target_int]:
                for i in range(len(left)):
                    if lmask & (1 << i):
                        v = others[i]
                        v_idx = elements.index(v)
                        conflict[idx_u][v_idx] = True
                        conflict[v_idx][idx_u] = True
        
        # Left + Right
        for rs in right_sums:
            if rs > target_int:
                continue
            need = target_int - rs
            if need in left_sums:
                for lmask in left_sums[need]:
                    for i in range(len(left)):
                        if lmask & (1 << i):
                            v = others[i]
                            v_idx = elements.index(v)
                            conflict[idx_u][v_idx] = True
                            conflict[v_idx][idx_u] = True
                for rmask in right_sums[rs]:
                    for i in range(len(right)):
                        if rmask & (1 << i):
                            v = others[half + i]
                            v_idx = elements.index(v)
                            conflict[idx_u][v_idx] = True
                            conflict[v_idx][idx_u] = True
    
    return conflict

def _mis_branch_and_bound(conflict, n, deadline):
    """
    Maximum Independent Set via branch-and-bound.
    conflict: n x n boolean matrix
    Returns the set of indices in the maximum independent set.
    """
    # Precompute neighbor sets
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflict[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    best_set = set()
    best_size = 0
    
    # Greedy initial solution
    remaining = set(range(n))
    greedy_set = set()
    while remaining:
        # Pick vertex with fewest neighbors in remaining
        v = min(remaining, key=lambda x: len(neighbors[x] & remaining))
        greedy_set.add(v)
        remaining -= neighbors[v]
        remaining.discard(v)
    
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set
    
    # Branch and bound
    def _bbs(candidates, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: |current_set| + |candidates|
        if len(current_set) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Pick a vertex to branch on
        v = next(iter(candidates))
        
        # Branch 1: Include v
        new_candidates = candidates - neighbors[v] - {v}
        new_current = current_set | {v}
        _bbs(new_candidates, new_current)
        
        # Branch 2: Exclude v
        new_candidates = candidates - {v}
        _bbs(new_candidates, current_set)
    
    _bbs(set(range(n)), set())
    
    return best_set

def _is_valid_set(S):
    """
    Check if set S is valid: no reciprocal of a selected element equals the sum
    of reciprocals of a subset of the OTHER selected elements.
    """
    S = sorted(S)
    n = len(S)
    
    # Compute LCM of all elements
    lcm_val = _lcm_list(S)
    
    for idx, x in enumerate(S):
        # Target: 1/x scaled by LCM = lcm_val // x
        target_int = lcm_val // x
        # Values: 1/y scaled by LCM = lcm_val // y for y != x
        values_int = [lcm_val // y for i, y in enumerate(S) if i != idx]
        
        # Check if target can be formed using meet-in-the-middle
        m = len(values_int)
        if m == 0:
            continue
        
        half = m // 2
        left = values_int[:half]
        right = values_int[half:]
        
        left_sums = set()
        left_sums.add(0)
        for mask in range(1, 1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            if s <= target_int:
                left_sums.add(s)
        
        right_sums = set()
        right_sums.add(0)
        for mask in range(1, 1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if s <= target_int:
                right_sums.add(s)
        
        if target_int in right_sums:
            return False
        if target_int in left_sums:
            return False
        for rs in right_sums:
            if rs > target_int:
                continue
            need = target_int - rs
            if need in left_sums:
                return False
    
    return True

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin
    
    # Build conflict graph
    conflict = _build_conflict_graph()
    
    # Find maximum independent set
    mis_indices = _mis_branch_and_bound(conflict, 24, deadline)
    
    # Convert to actual elements
    if mis_indices:
        result = sorted([i + 1 for i in mis_indices])
    else:
        # Fallback to known good set
        result = list(range(13, 25))
    
    # Verify and potentially improve with local search
    if time.time() < deadline - 5:
        best = set(result)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add an element
            for c in range(1, 25):
                if time.time() > deadline - 2:
                    break
                if c in best:
                    continue
                trial = best | {c}
                if _is_valid_set(trial):
                    best = trial
                    improved = True
                    break
        
        if len(best) > len(result):
            result = sorted(best)
    
    return result
