import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Precompute conflicts for each x
    # conflicts[x] is a list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1] # 0-indexed
            # For each element in the AP, the other three are its conflicts
            for i in range(4):
                x = ap[i]
                others = [ap[j] for j in range(4) if j != i]
                conflicts[x].append(tuple(others))

    def can_add(in_S, x):
        """
        Check if adding x to the set represented by in_S creates a 4-AP.
        x is 0-indexed.
        """
        for a, b, c in conflicts[x]:
            if in_S[a] and in_S[b] and in_S[c]:
                return False
        return True

    def greedy_construct(order):
        in_S = np.zeros(N, dtype=bool)
        count = 0
        for x in order:
            if can_add(in_S, x):
                in_S[x] = True
                count += 1
        return in_S, count

    best_in_S = np.zeros(N, dtype=bool)
    best_count = 0
    
    # Run multiple randomized greedy restarts
    while time.time() - start_time < time_limit:
        # Random order
        order = list(range(N))
        random.shuffle(order)
        
        in_S, count = greedy_construct(order)
        
        if count > best_count:
            best_count = count
            best_in_S = in_S.copy()

    # Convert best_in_S to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_in_S[i]]
    
    return result
