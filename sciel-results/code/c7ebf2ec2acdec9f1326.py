import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s buffer for safety
    
    # Pre-compute potential 4-APs? 
    # Or just check on the fly.
    # On the fly is fast enough.
    
    best_size = 0
    best_set = set()
    
    # We can also try a deterministic greedy first to have a baseline
    # But randomized is better.
    
    # Helper to check if adding x to S creates a 4-AP
    def can_add(x, S):
        # x is 1-indexed in problem, but we can use 0-indexed internally for speed if needed, 
        # but set lookup is O(1) so 1-indexed is fine.
        
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        # x+3d <= N
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return False
        
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        # x-d >= 1 and x+2d <= N
        # d <= x-1 and d <= (N-x)//2
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return False
        
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        # x-2d >= 1 and x+d <= N
        # d <= (x-1)//2 and d <= N-x
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return False
        
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        # x-3d >= 1
        # d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return False
        
        return True

    # Randomized Greedy
    # We run many iterations
    # To make it faster, we can use a list for S and a set for O(1) lookup, 
    # but we only need the set for lookup. The list is for iteration if we do local search.
    # Here we just build the set.
    
    # Optimization: Instead of checking all d every time, we can pre-compute conflicts?
    # No, S changes.
    
    # Let's try to run as many random restarts as possible.
    
    # One improvement: Sort the shuffled list? No, shuffle is key.
    
    # We can also try "greedy by frequency" or other heuristics, but random shuffle is robust.
    
    # Let's add a simple local search step if time permits?
    # For now, just random restarts.
    
    # To ensure we return a valid non-empty set even if something goes wrong, 
    # we start with a trivial set like {1}.
    
    current_best = {1}
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        current_set = set()
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(current_best):
            current_best = current_set.copy()
            
    return sorted(current_best)
