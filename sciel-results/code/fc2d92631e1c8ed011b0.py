import random
import time
import math
from math import gcd

def construct():
    N = 128
    nums = list(range(1, N+1))
    
    # Precompute GCD matrix
    G = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    def is_valid_add(x, S_list, in_S):
        # Check if x can be added to S_list without creating a bad triple
        # in_S is a boolean list of size N+1
        groups = {}
        for y in S_list:
            g = G[x][y]
            if g not in groups:
                groups[g] = []
            groups[g].append(y)
        
        for g, members in groups.items():
            if len(members) < 2:
                continue
            # Check all pairs in members
            for i in range(len(members)):
                a = members[i]
                for j in range(i+1, len(members)):
                    b = members[j]
                    if G[a][b] == g:
                        return False
        return True

    # Initial set using greedy
    def greedy_init():
        best_S = []
        rng = random.Random(42)
        in_S = [False]*(N+1)
        for _ in range(20):
            order = list(nums)
            rng.shuffle(order)
            S = []
            in_S = [False]*(N+1)
            for x in order:
                if in_S[x]:
                    continue
                if is_valid_add(x, S, in_S):
                    S.append(x)
                    in_S[x] = True
            if len(S) > len(best_S):
                best_S = S
        return best_S

    S = greedy_init()
    in_S = [False]*(N+1)
    for x in S:
        in_S[x] = True
        
    current_size = len(S)
    best_S = list(S)
    best_size = current_size
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 55.0 # Leave 5s for safety
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    rng = random.Random(123)
    
    # Precompute candidates list for speed? No, S changes.
    # But we can iterate 1..N and check in_S.
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random element not in S
            # Find all not in S
            candidates = [x for x in nums if not in_S[x]]
            if not candidates:
                continue
            x_in = rng.choice(candidates)
            if is_valid_add(x_in, S, in_S):
                S.append(x_in)
                in_S[x_in] = True
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = list(S)
                # Always accept valid add
            else:
                # Invalid, do nothing
                pass
        elif move_type < 0.7:
            # Try to remove a random element from S
            if not S:
                continue
            x_out = rng.choice(S)
            S.remove(x_out)
            in_S[x_out] = False
            current_size -= 1
            # Accept removal with probability
            delta_E = -1
            if T > 0:
                p = 1.0 / (1.0 + math.exp(-delta_E / T))
                if rng.random() > p:
                    S.append(x_out)
                    in_S[x_out] = True
                    current_size += 1
        else:
            # Try to swap
            if not S:
                continue
            x_out = rng.choice(S)
            candidates = [x for x in nums if not in_S[x]]
            if not candidates:
                continue
            x_in = rng.choice(candidates)
            
            # Check if swap is valid
            # Temporarily remove x_out
            S_temp = [y for y in S if y != x_out]
            if is_valid_add(x_in, S_temp, in_S):
                # Perform swap
                S.remove(x_out)
                in_S[x_out] = False
                S.append(x_in)
                in_S[x_in] = True
                # Size unchanged
                # Always accept swap to explore
                pass

        T *= cooling_rate

    return best_S
