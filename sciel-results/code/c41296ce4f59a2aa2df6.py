"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit
    
    best_set = []
    
    def is_valid_addition(S, x, sums_set):
        """Check if adding x to S keeps all triple sums unique."""
        # New triple sums involving x:
        # (x, x, x), (x, x, y) for y in S, (x, y, z) for y <= z in S
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in sums_set:
            return False
        new_sums.add(s)
        
        # (x, x, y) for y in S
        for y in S:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x, y, z) for y <= z in S (unordered, so y <= z)
        for i in range(len(S)):
            y = S[i]
            for j in range(i, len(S)):
                z = S[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False
                new_sums.add(s)
        
        return True
    
    def run_greedy(start_val=1, end_val=4097, seed=0):
        """Greedy construction with incremental validity checking."""
        rng = random.Random(seed)
        S = []
        sums_set = set()
        
        # Generate candidates in order
        candidates = list(range(start_val, end_val))
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if is_valid_addition(S, x, sums_set):
                # Add x to S and update sums_set
                S.append(x)
                # Add all new triple sums involving x
                # (x, x, x)
                sums_set.add(3 * x)
                # (x, x, y) for y in S[:-1]
                for y in S[:-1]:
                    sums_set.add(2 * x + y)
                # (x, y, z) for y <= z in S[:-1]
                for i in range(len(S) - 1):
                    y = S[i]
                    for j in range(i, len(S) - 1):
                        z = S[j]
                        sums_set.add(x + y + z)
        
        return S
    
    # Try multiple seeds/orders to find a better set
    best_len = 0
    best_result = []
    
    # First, try the natural greedy order
    result = run_greedy(1, 4097, 0)
    if len(result) > best_len:
        best_len = len(result)
        best_result = result
    
    # Try different random orderings if time permits
    if time.time() < deadline - 5:
        for seed in range(1, 50):
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed)
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
            
            S = []
            sums_set = set()
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                
                if is_valid_addition(S, x, sums_set):
                    S.append(x)
                    sums_set.add(3 * x)
                    for y in S[:-1]:
                        sums_set.add(2 * x + y)
                    for i in range(len(S) - 1):
                        y = S[i]
                        for j in range(i, len(S) - 1):
                            z = S[j]
                            sums_set.add(x + y + z)
            
            if len(S) > best_len:
                best_len = len(S)
                best_result = S
    
    return best_result if best_result else [1, 2, 4]
# EVOLVE-BLOCK-END
