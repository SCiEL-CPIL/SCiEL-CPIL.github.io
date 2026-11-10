import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Helper to check if x can be added to S without creating 4-AP
    def can_add(S, x):
        # x is 1st: x, x+d, x+2d, x+3d
        # x is 2nd: x-d, x, x+d, x+2d
        # x is 3rd: x-2d, x-d, x, x+d
        # x is 4th: x-3d, x-2d, x-d, x
        
        # Max d such that terms are in [1, N]
        # For x as 1st: x+3d <= N => d <= (N-x)//3
        # For x as 2nd: x-d >= 1 => d <= x-1; x+2d <= N => d <= (N-x)//2
        # For x as 3rd: x-2d >= 1 => d <= (x-1)//2; x+d <= N => d <= N-x
        # For x as 4th: x-3d >= 1 => d <= (x-1)//3
        
        # We can just iterate d from 1 to N//3 + 1 and check bounds
        # Or optimize ranges.
        
        # Case 1: x is start
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # Case 2: x is 2nd (x-d, x, x+d, x+2d)
        # d <= x-1 and x+2d <= N
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # Case 3: x is 3rd (x-2d, x-d, x, x+d)
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # Case 4: x is 4th (x-3d, x-2d, x-d, x)
        # x-3d >= 1 => d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        
        return True

    def greedy_fill(order, initial_set):
        S = set(initial_set)
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    # 1. Initial Best via several random greedy runs
    best_set = set()
    best_size = 0
    
    # Run a few random greedy starts
    for _ in range(50):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_fill(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Also try sorted greedy as a baseline
    S_sorted = greedy_fill(range(1, N + 1), set())
    if len(S_sorted) > best_size:
        best_size = len(S_sorted)
        best_set = S_sorted

    # 2. Local Search: Kick and Recover
    # We try to improve best_set by removing some elements and re-filling
    kick_sizes = [1, 2, 3, 4, 5]
    
    while time.time() < deadline:
        improved = False
        # Try different kick sizes
        for k in kick_sizes:
            if len(best_set) < k:
                continue
                
            # Try a few kicks of size k
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Pick k random elements to remove
                to_remove = random.sample(list(best_set), k)
                S_kicked = best_set - set(to_remove)
                
                # Candidates: all elements not in S_kicked
                # To make it faster, we can prioritize elements that were just removed
                # and other "promising" elements. 
                # But shuffling all 512 - len(S_kicked) is fast enough.
                candidates = [x for x in range(1, N + 1) if x not in S_kicked]
                random.shuffle(candidates)
                
                S_new = greedy_fill(candidates, S_kicked)
                
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best_set = S_new
                    improved = True
                    # Restart kick sizes or continue? Usually restart is good to exploit new basin
                    break 
            if improved:
                break
        if not improved:
            # If no improvement, maybe do a full random restart to escape deep local minimum
            # Or just continue with random kicks.
            # Let's do a full random greedy restart occasionally
            if random.random() < 0.1:
                order = list(range(1, N + 1))
                random.shuffle(order)
                S = greedy_fill(order, set())
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S

    return sorted(best_set)
