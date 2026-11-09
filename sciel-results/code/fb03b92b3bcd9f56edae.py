import time
from itertools import combinations

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # weights[i] = L // i, so 1/i = weights[i] / L
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    def is_valid_set(s):
        """Check if set s is reciprocal-subsum-free."""
        s_list = list(s)
        n = len(s_list)
        if n <= 1:
            return True
        
        # Precompute weights
        ws = [weights[x] for x in s_list]
        
        for i, x in enumerate(s_list):
            wx = ws[i]
            # Check if wx can be formed as subset sum of all other weights
            others = [ws[j] for j in range(n) if j != i]
            
            # Subset sum: can we form wx from others?
            # Use bitset-like approach with set of reachable sums
            reachable = {0}
            for w in others:
                if w > wx:
                    continue
                new_reachable = set()
                for r in reachable:
                    nr = r + w
                    if nr == wx:
                        return False
                    if nr < wx:
                        new_reachable.add(nr)
                reachable.update(new_reachable)
        
        return True
    
    # Strategy: use backtracking search with good ordering and pruning
    # Start with a greedy construction, then try to improve via local search
    
    # Greedy construction: add elements in decreasing order if valid
    def greedy_construct():
        result = set()
        # Try adding from largest to smallest (larger numbers have smaller reciprocals,
        # so they're less likely to be subsummed)
        for x in range(24, 0, -1):
            trial = result | {x}
            if is_valid_set(trial):
                result = trial
        return result
    
    # Local improvement: try to add elements, and if stuck, try swap operations
    def local_improve(s, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            s_list = list(s)
            
            # Try adding each missing element
            for y in range(24, 0, -1):
                if y in s:
                    continue
                trial = s | {y}
                if is_valid_set(trial):
                    s = trial
                    improved = True
                    break
            
            # If no single addition works, try swaps: remove one, add one
            if not improved and time.time() < deadline:
                for remove_x in s_list:
                    for add_y in range(24, 0, -1):
                        if add_y in s:
                            continue
                        trial = (s - {remove_x}) | {add_y}
                        if len(trial) > len(s):
                            if is_valid_set(trial):
                                s = trial
                                improved = True
                                break
                    if improved:
                        break
        
        return s
    
    # Backtracking search for optimal or near-optimal set
    def backtrack_search(deadline):
        # Order elements: try larger elements first (they have smaller reciprocals)
        elements = list(range(24, 0, -1))
        n = len(elements)
        
        best_set = set()
        best_size = 0
        
        # Precompute: for each element, its weight
        el_weights = [weights[e] for e in elements]
        
        # For pruning: max possible additional elements from position i onward
        # We use a simple bound: n - idx
        # Better bound: we can't add more than the number of remaining elements
        
        def backtrack(idx, selected, selected_set):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
            
            # Update best
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected_set)
            
            # Pruning: can we beat best_size?
            if len(selected) + (n - idx) <= best_size:
                return
            
            if idx >= n:
                return
            
            x = elements[idx]
            
            # Try including x
            trial_set = selected_set | {x}
            if is_valid_set(trial_set):
                backtrack(idx + 1, selected + [x], trial_set)
            
            # Try excluding x
            backtrack(idx + 1, selected, selected_set)
        
        backtrack(0, [], set())
        return best_set
    
    # Start with greedy
    best = greedy_construct()
    
    # Try local improvement
    if time.time() < deadline:
        best = local_improve(best, deadline)
    
    # If we have time, try backtracking search (but it may be slow for size 24)
    # Only try if greedy didn't find a large set
    if time.time() < deadline - 5 and len(best) < 18:
        bt_result = backtrack_search(deadline)
        if len(bt_result) > len(best):
            best = bt_result
            # Further local improve
            if time.time() < deadline:
                best = local_improve(best, deadline)
    
    # Final local improvement pass
    if time.time() < deadline:
        best = local_improve(best, deadline)
    
    if not best:
        best = {24}
    
    return list(best)
