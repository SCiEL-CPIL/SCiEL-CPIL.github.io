import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    def can_add(S, x):
        # x is 1st: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # x is 2nd: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # x is 3rd: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # x is 4th: x-3d, x-2d, x-d, x
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

    def get_cantor_set(n):
        """
        Returns the set of integers in [1, n] with no digit '2' in their ternary representation.
        This set is 3-AP free, and thus 4-AP free.
        """
        S = set()
        # Max ternary digits needed for n
        # 3^6 = 729 > 512
        # So we consider up to 6 digits (positions 0..5)
        max_power = 0
        temp = n
        while temp > 0:
            max_power += 1
            temp //= 3
        
        # Generate all combinations of 0/1 for these digits
        # But we must ensure the number is <= n and >= 1
        # We can just iterate 0..n and check? 512 is small.
        # Or generate recursively.
        # Iterating 1..n is O(N log N) which is fast.
        for i in range(1, n + 1):
            x = i
            valid = True
            while x > 0:
                if x % 3 == 2:
                    valid = False
                    break
                x //= 3
            if valid:
                S.add(i)
        return S

    # 1. Initial Best via several random greedy runs AND Cantor set based run
    best_set = set()
    best_size = 0
    
    # Run Cantor set based greedy
    cantor_base = get_cantor_set(N)
    # To allow greedy to add more, we need an order.
    # If we start with Cantor set, we try to add others.
    # The order matters. Random order is good.
    for _ in range(10): # Fewer runs for this specific strategy, as it's structured
        if time.time() > deadline: break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_cantor = greedy_fill(order, cantor_base)
        if len(S_cantor) > best_size:
            best_size = len(S_cantor)
            best_set = S_cantor

    # Run standard random greedy starts
    for _ in range(50):
        if time.time() > deadline: break
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
    kick_sizes = [1, 2, 3, 4, 5]
    
    while time.time() < deadline:
        improved = False
        for k in kick_sizes:
            if len(best_set) < k:
                continue
                
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                to_remove = random.sample(list(best_set), k)
                S_kicked = best_set - set(to_remove)
                
                candidates = [x for x in range(1, N + 1) if x not in S_kicked]
                random.shuffle(candidates)
                
                S_new = greedy_fill(candidates, S_kicked)
                
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best_set = S_new
                    improved = True
                    break 
            if improved:
                break
        if not improved:
            if random.random() < 0.1:
                order = list(range(1, N + 1))
                random.shuffle(order)
                S = greedy_fill(order, set())
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S

    return sorted(best_set)
