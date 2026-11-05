import time
import random
from math import gcd

N = 512
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # Precompute compatibility
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        compat[i][i] = True
        for j in range(i + 1, N + 1):
            val = i // gcd(i, j) * j
            compat[i][j] = val <= N
            compat[j][i] = compat[i][j]
    
    def can_add_to(S, x):
        for y in S:
            if not compat[x][y]:
                return False
        return True
    
    def greedy(order):
        S = set()
        for x in order:
            if can_add_to(S, x):
                S.add(x)
        return S
    
    best_set = set()
    
    orders = [
        list(range(1, N + 1)),
        list(range(1, N + 1))[::-1],
        sorted(range(1, N + 1), key=lambda x: x.bit_length()),
        sorted(range(1, N + 1), key=lambda x: -x.bit_count()),
    ]
    
    for order in orders:
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S
    
    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            # Try to add any compatible element
            for x in range(1, N + 1):
                if x not in S and can_add_to(S, x):
                    S.add(x)
                    improved = True
            # Try 1-for-1 swaps that might enable growth
            if not improved:
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    for y in list(S):
                        S2 = S - {y}
                        if can_add_to(S2, x):
                            # Check if this enables adding more
                            S = S2
                            S.add(x)
                            # Try to add more
                            added = False
                            for z in range(1, N + 1):
                                if z not in S and can_add_to(S, z):
                                    S.add(z)
                                    added = True
                            if added:
                                improved = True
                            else:
                                # Revert if no improvement
                                S.discard(x)
                                S.add(y)
                            break
                    if improved:
                        break
        return S
    
    best_set = local_search(best_set)
    
    # Random restarts
    if time.time() - start_time < DEADLINE - 10:
        for trial in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            seed_size = random.randint(3, 25)
            random_seed = random.sample(range(1, N + 1), seed_size)
            valid_seed = set()
            for x in random_seed:
                if can_add_to(valid_seed, x):
                    valid_seed.add(x)
            S = local_search(valid_seed)
            if len(S) > len(best_set):
                best_set = S
    
    return list(best_set)

