import time
from fractions import Fraction

def construct():
    start_time = time.time()
    best_set = []
    best_len = 0
    
    def dfs(next_u, current_set, subset_sums):
        nonlocal best_set, best_len
        
        # Time check
        if time.time() - start_time > 85:
            return

        # Pruning
        if len(current_set) + next_u <= best_len:
            return
        
        if next_u == 0:
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set.copy()
            return
        
        target = Fraction(1, next_u)
        
        # Try including next_u
        if target not in subset_sums:
            new_set = current_set + [next_u]
            # Optimization: only compute new sums if we actually recurse?
            # Yes.
            new_sums = set(subset_sums)
            for s in subset_sums:
                new_sums.add(s + target)
            dfs(next_u - 1, new_set, new_sums)
        
        # Try excluding next_u
        dfs(next_u - 1, current_set, subset_sums)

    dfs(24, [], {Fraction(0)})
    
    if not best_set:
        return [1]
        
    return best_set
