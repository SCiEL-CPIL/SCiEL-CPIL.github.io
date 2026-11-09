import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Helper: Check if adding x to S creates a 4-AP
    # S is a set of integers
    def can_add(S, x):
        # x is 1st term: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        # x is 2nd term: x-d, x, x+d, x+2d
        # x-d >= 1 => d <= x-1
        # x+2d <= N => d <= (N-x)//2
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # x is 3rd term: x-2d, x-d, x, x+d
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # x is 4th term: x-3d, x-2d, x-d, x
        # x-3d >= 1 => d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Run multiple random restarts
    while time.time() - start_time < time_limit:
        S = set()
        # Random order for greedy
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Try to improve by removing one element and re-greedy
        # This helps escape local optima
        if len(S) > 0:
            # Pick a random element to remove
            # Maybe remove one that is "central" or just random
            remove_x = random.choice(list(S))
            S.remove(remove_x)
            
            # Try to add back elements from the remaining pool
            # We need to track which elements are not in S
            # Simpler: just run a few more greedy passes on the complement?
            # Or just do a single pass of adding any element that fits
            # This is essentially a local search step
            
            # Let's do a quick pass to see if we can add something back
            # Actually, the main loop above does a full greedy.
            # A more effective local search:
            # 1. Remove random x
            # 2. Try to add all other elements not in S in random order
            # 3. If size increases, keep it.
            
            # To save time, we'll just do the main restart loop.
            # But we can add a small local search phase here.
            
            # Let's stick to the restart loop for simplicity and speed.
            pass

    return sorted(best_set)
