import time
import random

def _count_conflicts(S):
    """
    Count the number of sums that appear in more than one cardinality.
    Returns 0 if the set is valid.
    """
    n = len(S)
    if n == 0:
        return 0
    
    # seen[s] = cardinality if s was seen with that cardinality, -1 if conflict
    seen = {}
    conflicts = 0
    
    # Iterate through all subsets
    # 2^n subsets
    for mask in range(1 << n):
        # Calculate sum and cardinality
        s = 0
        c = 0
        # Optimization: use bit operations? 
        # For n <= 12, simple loop is fine.
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
                c += 1
        
        if s in seen:
            if seen[s] != -1 and seen[s] != c:
                # Conflict: s was seen with a different cardinality
                if seen[s] != c:
                    conflicts += 1
                    seen[s] = -1
            # If seen[s] == c, it's a repeat within same cardinality, allowed.
            # If seen[s] == -1, it's already a conflict, don't double count.
        else:
            seen[s] = c
            
    return conflicts

def _is_valid(S):
    return _count_conflicts(S) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Fallback: Powers of 2
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    best_size = 9
    best_set = fallback
    
    # Try to find larger sets
    # Max possible size is bounded by 2^k <= k*256+1
    # k=12 -> 4096 <= 3073 (False)
    # So max size is likely 11 or less?
    # Wait, the bound 2^k <= k*256+1 is for ALL sums being distinct.
    # We only need disjointness between cardinalities.
    # The total number of distinct sums required is sum_{i=0}^k |S_i|.
    # |S_i| can be less than C(k,i) if there are internal collisions.
    # But usually |S_i| is close to C(k,i).
    # So 2^k is a good upper bound on the number of distinct sums needed.
    # The range of sums is 0 to k*256.
    # So we need roughly 2^k <= k*256.
    # k=11: 2048 <= 2816. OK.
    # k=12: 4096 <= 3072. NOT OK.
    # So size 12 is very unlikely/impossible.
    # Let's try 11 and 10.
    
    for target_size in [11, 10]:
        if time.time() > deadline:
            break
            
        # Randomized Local Search
        # We will run multiple restarts
        max_restarts = 50
        restarts = 0
        
        while restarts < max_restarts and time.time() < deadline:
            # Initialize random set of target_size
            S = random.sample(range(1, 257), target_size)
            S.sort()
            
            current_conflicts = _count_conflicts(S)
            
            if current_conflicts == 0:
                return S
            
            # Local Search
            no_improve_count = 0
            max_no_improve = 100
            
            while no_improve_count < max_no_improve and time.time() < deadline:
                # Pick a random index to swap
                i = random.randint(0, target_size - 1)
                # Pick a random new value
                new_val = random.randint(1, 256)
                
                # Ensure distinctness
                if new_val in S:
                    # Try a few times to get a distinct value
                    for _ in range(10):
                        new_val = random.randint(1, 256)
                        if new_val not in S:
                            break
                    else:
                        continue
                
                old_val = S[i]
                S[i] = new_val
                S.sort()
                
                new_conflicts = _count_conflicts(S)
                
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    no_improve_count = 0
                    
                    if current_conflicts == 0:
                        return S
                else:
                    # Accept with some probability (Simulated Annealing style)
                    # Or just revert if not improving, to keep it simple
                    # Let's try a simple acceptance: accept if equal, or with small prob if worse
                    if new_conflicts == current_conflicts:
                        no_improve_count += 1
                    else:
                        # Revert
                        S[i] = old_val
                        S.sort()
                        no_improve_count += 1
                        
                if no_improve_count >= max_no_improve:
                    break
            
            restarts += 1
            
        # If we exited the loop for this target_size, continue to next smaller size
    
    return fallback
