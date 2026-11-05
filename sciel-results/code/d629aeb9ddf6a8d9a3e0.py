import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 4096
    best_set = []
    best_size = 0

    def try_add(S, c, used_sums):
        # Returns True if c can be added, and updates used_sums if so?
        # Better to separate check and update to avoid side effects if check fails?
        # Or just return the new sums.
        
        new_sums = []
        # 3c
        s_val = 3 * c
        if s_val in used_sums:
            return None
        new_sums.append(s_val)

        # 2c + s
        for s in S:
            s_val = 2 * c + s
            if s_val in used_sums:
                return None
            new_sums.append(s_val)

        # c + s1 + s2
        for i in range(len(S)):
            s1 = S[i]
            for j in range(i, len(S)):
                s2 = S[j]
                s_val = c + s1 + s2
                if s_val in used_sums:
                    return None
                new_sums.append(s_val)
                
        # Check internal duplicates
        if len(new_sums) != len(set(new_sums)):
            return None
            
        return new_sums

    def add_to_set(S, c, used_sums, new_sums):
        S.append(c)
        used_sums.update(new_sums)

    # 1. Deterministic Greedy
    S = []
    used_sums = set()
    for c in range(1, n + 1):
        new_sums = try_add(S, c, used_sums)
        if new_sums is not None:
            add_to_set(S, c, used_sums, new_sums)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        if time.time() > deadline:
            return best_set

    # 2. Randomized Greedy
    # We will run as many as possible
    # To speed up, we can skip candidates that are "too similar" to existing ones?
    # No, just shuffle.
    
    # We can also try different random seeds or strategies
    # Strategy: Shuffle [1, 4096]
    
    while time.time() < deadline - 5:
        S = []
        used_sums = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            # Optional: if len(S) > best_size, we are already improving
            # but we should still try to maximize this specific run
            new_sums = try_add(S, c, used_sums)
            if new_sums is not None:
                add_to_set(S, c, used_sums, new_sums)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    
        # If we found a very good set, maybe break?
        # Theoretical max is ~42. If we get 40, it's hard to beat.
        # But we don't know the exact max.
        
    return best_set
