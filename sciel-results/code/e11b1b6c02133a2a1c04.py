import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    max_time = 55  # Leave some buffer for 60s limit
    
    # Use a bytearray for S, it's compact and fast
    # But we need to reset it each time.
    
    while time.time() - start_time < max_time:
        S = bytearray(N + 1)
        count = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if S[x]:
                continue
                
            valid = True
            S_local = S # Local reference
            
            # Check if x is the 1st term
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if S_local[x + d] and S_local[x + 2 * d] and S_local[x + 3 * d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 2nd term
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if S_local[x - d] and S_local[x + d] and S_local[x + 2 * d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 3rd term
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if S_local[x - 2 * d] and S_local[x - d] and S_local[x + d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 4th term
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if S_local[x - 3 * d] and S_local[x - 2 * d] and S_local[x - d]:
                    valid = False
                    break
            
            if valid:
                S[x] = 1
                count += 1
        
        if count > len(best_set):
            # Convert bytearray to set for comparison
            best_set = {i for i in range(1, N + 1) if S[i]}
            
    return sorted(best_set)
