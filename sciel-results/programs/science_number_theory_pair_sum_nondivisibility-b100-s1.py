import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute conflict matrix and conflict lists
    n = 256
    C = [[False] * (n + 1) for _ in range(n + 1)]
    conflicts = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                C[i][j] = True
                C[j][i] = True
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Helper: Check if x can be added to set S
    def can_add(x, S):
        # x can be added if no element in S conflicts with x
        # i.e., S ∩ conflicts[x] is empty
        return len(S & conflicts[x]) == 0
        
    # Initial solution: All odd numbers
    # Odd numbers: a+b is even, ab is odd. Even doesn't divide odd. So no conflicts.
    S = set(range(1, n + 1, 2))
    best_S = set(S)
    best_size = len(best_S)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.005
    
    # Precompute list of numbers for random selection
    all_nums = list(range(1, n + 1))
    
    # Track stagnation for restarts
    stagnation_count = 0
    stagnation_threshold = 500
    restart_T = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a move
        r = random.random()
        
        # Try to Add
        if r < 0.5:
            x = random.choice(all_nums)
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1
        # Try to Remove
        elif r < 0.85:
            if S:
                u = random.choice(list(S))
                # Accept removal with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    S.remove(u)
                    stagnation_count += 1
        # Try to Swap
        else:
            if S:
                u = random.choice(list(S))
                x = random.choice(all_nums)
                if x not in S:
                    # Check if x is compatible with S \ {u}
                    # x conflicts with S minus {u} only if (S \ {u}) ∩ conflicts[x] is non-empty
                    # Equivalently: (S ∩ conflicts[x]) ⊆ {u}
                    # i.e., S ∩ conflicts[x] is either empty or exactly {u}
                    inter = S & conflicts[x]
                    if not inter or (inter == {u}):
                        # Size remains same. Accept to explore
                        if random.random() < 0.5:
                            S.remove(u)
                            S.add(x)
                            stagnation_count += 1
        
        # Restart if stuck
        if stagnation_count > stagnation_threshold:
            T = restart_T
            stagnation_count = 0
        
        # Cool down
        T *= cooling
        
    return list(best_S)
