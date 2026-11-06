import math
import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i (integer representation of 1/i scaled by L)
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    def subset_sum_exists(target, values):
        """Check if any subset of values sums to target.
        values is a list of integers.
        Returns True/False.
        """
        if target < 0:
            return False
        if not values:
            return target == 0
        total = sum(values)
        if total < target:
            return False
        if target == 0:
            return True
        
        k = len(values)
        if k <= 1:
            return any(v == target for v in values)
        
        # Meet in the middle
        half = k // 2
        left = values[:half]
        right = values[half:]
        
        # Generate all subset sums of left
        left_sums = {0}
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        
        # Check if any right subset sum completes a left sum to target
        right_sums = {0}
        for v in right:
            new_sums = {s + v for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    return True
            right_sums = right_sums | new_sums
        
        return False

    def is_valid_incremental(S_list, v):
        """
        Check if adding v to S_list (which is valid) keeps it valid.
        S_list is a list of elements (1-24).
        v is an element (1-24) not in S_list.
        """
        # 1. Check if 1/v is sum of subset of S
        s_values = [V[i] for i in S_list]
        if subset_sum_exists(V[v], s_values):
            return False
        
        # 2. Check for each x in S, if 1/x - 1/v is sum of subset of S \ {x}
        # i.e., V[x] - V[v] is sum of subset of S \ {x}
        for x in S_list:
            target = V[x] - V[v]
            if target <= 0:
                continue
            # Values available: S \ {x}
            avail_values = [V[i] for i in S_list if i != x]
            if subset_sum_exists(target, avail_values):
                return False
                
        return True

    def is_valid_full(S_list):
        """Check if S_list is valid from scratch."""
        if not S_list:
            return True
        S_set = set(S_list)
        for x in S_list:
            other_elements = [i for i in S_list if i != x]
            other_values = [V[i] for i in other_elements]
            if subset_sum_exists(V[x], other_values):
                return False
        return True

    # Phase 1: Start with a good initial solution
    # Heuristic: larger elements have smaller reciprocals, less likely to be subsums
    # Try starting with top half
    best = list(range(13, 25))
    
    # Greedy improvement: try adding elements
    all_elems = list(range(1, 25))
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for v in all_elems:
            if (time.time() - start_time) > deadline:
                break
            if v in best:
                continue
            if is_valid_incremental(best, v):
                best.append(v)
                improved = True
                break
    
    best_size = len(best)
    best_set = set(best)
    
    # Phase 2: Local search with remove-1-add-2 moves
    if (time.time() - start_time) < deadline:
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            best_set = set(best)
            
            # Try remove-1-add-1
            for i in range(len(best)):
                if (time.time() - start_time) > deadline:
                    break
                for j in range(1, 25):
                    if (time.time() - start_time) > deadline:
                        break
                    if j in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j]
                    if len(candidate) > len(best) and is_valid_full(candidate):
                        best = candidate
                        best_set = set(best)
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            
            if improved or (time.time() - start_time) > deadline:
                continue
            
            # Try remove-1-add-2
            for i in range(len(best)):
                if (time.time() - start_time) > deadline:
                    break
                for j, k in combinations(range(1, 25), 2):
                    if (time.time() - start_time) > deadline:
                        break
                    if j in best_set or k in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j, k]
                    if len(candidate) > len(best) and is_valid_full(candidate):
                        best = candidate
                        best_set = set(best)
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
    
    # Phase 3: Branch and bound to try to find larger sets
    if (time.time() - start_time) < deadline - 10:  # Need at least 10s for B&B
        # Order elements: try larger elements first (smaller reciprocals)
        order = sorted(range(1, 25), key=lambda x: -x)
        
        def dfs(current_set, idx, current_size):
            nonlocal best, best_size, best_set
            
            if (time.time() - start_time) > deadline:
                return
            
            # Pruning: if we can't beat best, stop
            if current_size + (len(order) - idx) <= best_size:
                return
            
            if idx >= len(order):
                return
            
            elem = order[idx]
            
            # Try including elem
            if elem not in best_set:
                if is_valid_incremental(current_set, elem):
                    new_set = current_set + [elem]
                    dfs(new_set, idx + 1, current_size + 1)
            
            # Try excluding elem
            dfs(current_set, idx + 1, current_size)
        
        dfs(list(best), 0, len(best))
    
    # Final validation
    if not is_valid_full(best):
        # Fallback to known good
        best = list(range(13, 25))
    
    return sorted(best)
