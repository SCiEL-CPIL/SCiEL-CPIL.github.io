import numpy as np

def construct():
    n = 50
    best_set = [0]
    best_size = 1
    
    # Backtracking search for maximal Sidon set
    # We want to find a set S such that all sums i+j are unique (for i <= j)
    # Equivalent to: all differences i-j are unique (for i != j)
    
    used_sums = set()
    current_set = []
    
    def backtrack(start_idx, current_size):
        nonlocal best_set, best_size, used_sums, current_set
        
        # Pruning: if we can't beat the best size, return
        # Max possible size is roughly sqrt(n). 
        # Simple bound: current_size + (n - start_idx) <= best_size
        if current_size + (n - start_idx) <= best_size:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
            
        # Try adding indices from start_idx down to 0
        # Descending order might help spread them out
        for i in range(start_idx, -1, -1):
            # Check if i can be added
            valid = True
            new_sums = []
            
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
                
            if valid:
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
                else:
                    new_sums.append(s_ii)
                    
            if valid:
                # Add i
                current_set.append(i)
                used_sums.update(new_sums)
                
                # Recurse
                # Next index to try should be less than i to avoid duplicates in set
                backtrack(i - 1, current_size + 1)
                
                # Remove i
                current_set.pop()
                for s in new_sums:
                    used_sums.remove(s)
                    
        # Optional: Stop early if we found a very large set (e.g., 9 or 10)
        # But let's let it run to find the true max or a good one.
        # Given N=50, this should be fast.

    backtrack(n - 1, 0)
    
    h = np.zeros(n)
    k = len(best_set)
    if k == 0:
        h[0] = 1.0
    else:
        for idx in best_set:
            h[idx] = 1.0 / k
            
    return list(h)
