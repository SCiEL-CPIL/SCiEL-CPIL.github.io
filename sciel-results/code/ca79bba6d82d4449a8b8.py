import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    S = set(range(1, n + 1))
    
    # Memoize subset-sum results
    subset_sum_cache = {}
    
    def subset_sum_exists(target, values):
        """Check if any subset of values sums to target. Returns (bool, subset_indices)"""
        if not values:
            return (False, None)
        if sum(values) < target:
            return (False, None)
        
        key = (target, tuple(sorted(values)))
        if key in subset_sum_cache:
            return subset_sum_cache[key]
        
        # Meet-in-the-middle for up to ~20 elements, DFS otherwise
        if len(values) <= 20:
            # Meet in the middle
            half = len(values) // 2
            left = values[:half]
            right = values[half:]
            
            left_sums = {}
            for mask in range(1 << half):
                s = 0
                for i in range(half):
                    if mask & (1 << i):
                        s += left[i]
                left_sums[s] = mask
            
            right_sums = {}
            for mask in range(1 << (len(values) - half)):
                s = 0
                for i in range(len(values) - half):
                    if mask & (1 << i):
                        s += right[i]
                right_sums[s] = mask
            
            for s_l, mask_l in left_sums.items():
                need = target - s_l
                if need in right_sums:
                    mask_r = right_sums[need]
                    # Build the subset
                    subset = []
                    for i in range(half):
                        if mask_l & (1 << i):
                            subset.append(i)
                    for i in range(len(values) - half):
                        if mask_r & (1 << i):
                            subset.append(half + i)
                    result = (True, subset)
                    subset_sum_cache[key] = result
                    return result
            
            result = (False, None)
            subset_sum_cache[key] = result
            return result
        else:
            # DFS with pruning
            vals_sorted = sorted(values, reverse=True)
            prefix_sum = [0] * (len(vals_sorted) + 1)
            for i in range(len(vals_sorted)):
                prefix_sum[i + 1] = prefix_sum[i] + vals_sorted[i]
            
            def dfs(start, current, path):
                if current == target:
                    return path[:]
                if current > target:
                    return None
                if start >= len(vals_sorted):
                    return None
                if current + prefix_sum[len(vals_sorted)] - prefix_sum[start] < target:
                    return None
                
                # Include
                res = dfs(start + 1, current + vals_sorted[start], path + [start])
                if res is not None:
                    return res
                # Exclude
                res = dfs(start + 1, current, path)
                if res is not None:
                    return res
                return None
            
            res = dfs(0, 0, [])
            if res is not None:
                # Map back to original indices
                # We need to track which original value each sorted value came from
                # This is tricky; let's just return True
                result = (True, None)
            else:
                result = (False, None)
            subset_sum_cache[key] = result
            return result

    def find_conflict(x, S):
        """Find if any subset of S\{x} has reciprocal sum equal to 1/x.
        Returns list of elements in the conflicting subset, or None."""
        target = V[x]
        other_vals = [(V[y], y) for y in S if y != x]
        if not other_vals:
            return None
        
        vals = [v for v, _ in other_vals]
        if sum(vals) < target:
            return None
        
        # Use meet-in-the-middle
        if len(vals) <= 22:
            half = len(vals) // 2
            left_vals = vals[:half]
            right_vals = vals[half:]
            
            left_sums = {}
            for mask in range(1 << half):
                s = 0
                for i in range(half):
                    if mask & (1 << i):
                        s += left_vals[i]
                if s not in left_sums:
                    left_sums[s] = mask
            
            right_sums = {}
            for mask in range(1 << (len(vals) - half)):
                s = 0
                for i in range(len(vals) - half):
                    if mask & (1 << i):
                        s += right_vals[i]
                if s not in right_sums:
                    right_sums[s] = mask
            
            for s_l, mask_l in left_sums.items():
                need = target - s_l
                if need in right_sums:
                    mask_r = right_sums[need]
                    subset = []
                    for i in range(half):
                        if mask_l & (1 << i):
                            subset.append(other_vals[i][1])
                    for i in range(len(vals) - half):
                        if mask_r & (1 << i):
                            subset.append(other_vals[half + i][1])
                    return subset
            
            return None
        else:
            # DFS fallback
            vals_sorted = sorted(vals, reverse=True)
            # Track original indices
            sorted_pairs = sorted(zip(vals, [y for _, y in other_vals]), key=lambda p: -p[0])
            prefix_sum = [0] * (len(vals_sorted) + 1)
            for i in range(len(vals_sorted)):
                prefix_sum[i + 1] = prefix_sum[i] + vals_sorted[i]
            
            def dfs(start, current, path):
                if current == target:
                    return path[:]
                if current > target:
                    return None
                if start >= len(vals_sorted):
                    return None
                if current + prefix_sum[len(vals_sorted)] - prefix_sum[start] < target:
                    return None
                
                res = dfs(start + 1, current + vals_sorted[start], path + [start])
                if res is not None:
                    return res
                res = dfs(start + 1, current, path)
                if res is not None:
                    return res
                return None
            
            res = dfs(0, 0, [])
            if res is not None:
                return [sorted_pairs[i][1] for i in res]
            return None

    def is_valid(S):
        """Check if S is reciprocal-subsum-free."""
        for x in S:
            conflict = find_conflict(x, S)
            if conflict is not None:
                return False
        return True

    # Phase 1: Greedy removal
    while time.time() < deadline - 10.0:
        conflicts = []
        for x in S:
            conflict = find_conflict(x, S)
            if conflict is not None:
                conflicts.append((x, conflict))
        
        if not conflicts:
            break
        
        # Count how often each element appears in conflicts (as target or in witness)
        counts = {y: 0 for y in S}
        for x, witness in conflicts:
            counts[x] += 1
            for y in witness:
                counts[y] += 1
        
        # Remove the element with highest count
        best_elem = max(counts, key=lambda y: (counts[y], y))  # Tie-break: larger y
        S.remove(best_elem)
    
    # Phase 2: Try to add elements back
    removed = set(range(1, n + 1)) - S
    for y in sorted(removed, reverse=True):
        if time.time() > deadline - 5.0:
            break
        trial = S | {y}
        if is_valid(trial):
            S = trial
    
    # Phase 3: Local search - try remove-1-add-2 swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for x in list(S):
            for y1 in range(1, n + 1):
                if y1 in S:
                    continue
                for y2 in range(y1 + 1, n + 1):
                    if y2 in S:
                        continue
                    trial = (S - {x}) | {y1, y2}
                    if len(trial) > len(S) and is_valid(trial):
                        S = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Phase 4: Try single additions again
    removed = set(range(1, n + 1)) - S
    for y in sorted(removed, reverse=True):
        if time.time() > deadline - 2.0:
            break
        trial = S | {y}
        if is_valid(trial):
            S = trial
    
    return sorted(S)
