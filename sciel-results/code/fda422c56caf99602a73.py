import time
import random

START = time.time()
DEADLINE = 55.0

def _validate(vals):
    n = len(vals)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _solve_size(target_size):
    """
    Find a set of target_size distinct ints in 1..256 satisfying the constraint.
    Uses DFS with backtracking and bitsets.
    """
    available = list(range(1, 257))
    # Shuffle to allow different paths? No, let's try descending first for structure.
    # Actually, let's try a few orderings.
    
    def dfs(current_vals, current_sums, available_set):
        if time.time() - START > DEADLINE:
            return None
            
        if len(current_vals) == target_size:
            return list(current_vals)
            
        spots_left = target_size - len(current_vals)
        if len(available_set) < spots_left:
            return None
            
        # Try candidates
        # Heuristic: Try larger numbers first?
        # Sort available? That's expensive.
        # Let's just iterate.
        
        # To avoid modifying available_set in place in a way that breaks backtracking,
        # we create a new set or list.
        
        for c in available_set:
            # Check validity
            n = len(current_vals)
            valid = True
            new_sums = current_sums[:] # Copy
            
            # Compute new bits
            # We need to check if (current_sums[k] << c) intersects with any current_sums[m] where m != k+1
            for k in range(n + 1):
                if current_sums[k]:
                    new_bits = current_sums[k] << c
                    target_k = k + 1
                    for m in range(n + 1):
                        if m == target_k:
                            continue
                        if new_bits & current_sums[m]:
                            valid = False
                            break
                if not valid:
                    break
                    
            if valid:
                # Update sums
                for k in range(n, -1, -1):
                    if current_sums[k]:
                        new_sums[k+1] |= (current_sums[k] << c)
                        
                # Recurse
                new_available = available_set - {c}
                res = dfs(current_vals + [c], new_sums, new_available)
                if res:
                    return res
                    
        return None
        
    # Start DFS
    # We can try different initial orderings by shuffling available before starting?
    # The DFS iterates `available_set`. If it's a set, order is arbitrary.
    # Let's convert to list and sort descending for the first attempt.
    
    avail_list = list(range(256, 0, -1))
    # We can't easily pass a list to dfs and remove from it efficiently without copying.
    # Using a set is O(1) removal but iteration order is not controlled.
    # For speed, let's use a list and a "used" boolean array or set.
    
    # Revised DFS with list
    def dfs_list(idx, current_vals, current_sums, used):
        if time.time() - START > DEADLINE:
            return None
            
        if len(current_vals) == target_size:
            return list(current_vals)
            
        spots_left = target_size - len(current_vals)
        # Simple bound: if remaining elements in avail_list < spots_left, fail
        # We can precompute suffix counts or just check len(available_set) in the set version.
        
        for i in range(idx, len(avail_list)):
            c = avail_list[i]
            if used[c]:
                continue
                
            # Check validity
            n = len(current_vals)
            valid = True
            new_sums = current_sums[:]
            
            for k in range(n + 1):
                if current_sums[k]:
                    new_bits = current_sums[k] << c
                    target_k = k + 1
                    for m in range(n + 1):
                        if m == target_k:
                            continue
                        if new_bits & current_sums[m]:
                            valid = False
                            break
                if not valid:
                    break
                    
            if valid:
                for k in range(n, -1, -1):
                    if current_sums[k]:
                        new_sums[k+1] |= (current_sums[k] << c)
                        
                used[c] = True
                res = dfs_list(i + 1, current_vals + [c], new_sums, used)
                if res:
                    return res
                used[c] = False
                
        return None
        
    used = [False] * 257
    return dfs_list(0, [], [1] + [0]*target_size, used)

def construct():
    global START, DEADLINE
    
    # Try sizes 12 down to 1
    # 12 is a reasonable upper bound for this constraint with 256 max value.
    for size in range(12, 0, -1):
        if time.time() - START > DEADLINE:
            break
            
        result = _solve_size(size)
        if result:
            if _validate(result):
                return [int(x) for x in result]
                
    return [1, 2, 4, 8, 16, 32, 64, 128]
