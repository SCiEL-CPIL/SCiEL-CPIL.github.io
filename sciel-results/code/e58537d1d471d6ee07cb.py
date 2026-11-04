import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        
        return 1.0 + log_ratio / log_denom

    # Search for small k
    max_k = 8
    # Upper bound for max element M
    # Heuristic: M grows with k
    # For k=3, M=5
    # For k=4, M=10
    # For k=5, M=15
    # For k=6, M=25
    # For k=7, M=40
    # For k=8, M=60
    
    bounds = {3: 10, 4: 15, 5: 20, 6: 30, 7: 50, 8: 80}
    
    for k in range(3, max_k + 1):
        M = bounds.get(k, 100)
        
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
        # Generate combinations of size k-1 from 1..M
        # We must include 0
        candidates = range(1, M + 1)
        
        # To avoid excessive time for larger k, we can limit the search
        # or use a heuristic. But for k<=8 and M<=80, it might be too slow for k=8.
        # C(80, 7) is huge.
        # Let's restrict k to 6 for full search, and use greedy for 7,8?
        # Or just increase M slowly.
        
        if k > 6:
             # Use a greedy construction for larger k
             # Start with best set of size k-1 and try to add one element
             # This is complex to implement correctly in a short time.
             # Let's just search up to k=6 fully, and k=7 with smaller M.
             M = 30 
             
        # Full search for k <= 6
        # For k=7, M=30 is C(30, 6) ~ 600k. Fast enough.
        # For k=8, M=30 is C(30, 7) ~ 2M. Might be slow in Python.
        # Let's stick to k <= 7.
        
        if k > 7:
            continue
            
        for combo in itertools.combinations(candidates, k - 1):
            if time.time() > deadline - 2.0:
                return best_set
                
            U = [0] + list(combo)
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
