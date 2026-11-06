import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0
    
    S = set(range(1, 25))
    
    def find_conflict(x, S):
        candidates = [y for y in S if y > x]
        if not candidates:
            return None
        target = Fraction(1, x)
        
        # Sort by value ascending to prune faster (large reciprocals first)
        candidates.sort()
        
        total_sum = sum(Fraction(1, y) for y in candidates)
        if total_sum < target:
            return None
            
        subset_found = None
        
        def dfs(idx, current_sum, current_subset):
            nonlocal subset_found
            if subset_found is not None:
                return
            if current_sum == target:
                subset_found = list(current_subset)
                return
            if current_sum > target:
                return
            if idx == len(candidates):
                return
            
            # Include
            current_subset.append(candidates[idx])
            dfs(idx + 1, current_sum + Fraction(1, candidates[idx]), current_subset)
            if subset_found is not None:
                return
            current_subset.pop()
            
            # Exclude
            dfs(idx + 1, current_sum, current_subset)
            
        dfs(0, Fraction(0, 1), [])
        return subset_found

    while True:
        if time.time() > deadline:
            break
            
        removed = False
        # Iterate x in increasing order
        for x in sorted(S):
            subset = find_conflict(x, S)
            if subset is not None:
                # Conflict: 1/x = sum(1/u for u in subset)
                involved = [x] + subset
                to_remove = min(involved)
                S.remove(to_remove)
                removed = True
                break
                
        if not removed:
            break
            
    return list(S)
