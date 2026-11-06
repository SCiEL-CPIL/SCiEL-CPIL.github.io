"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # 2. Precompute conflict masks for pairs
    # conflict_mask[i][j] is a bitmask of all k such that (i,j,k) is a bad triple
    conflict_mask = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd_table[i][j]
            mask = 0
            # Iterate k to find conflicts
            for k in range(1, N+1):
                if k == i or k == j:
                    continue
                if gcd_table[i][k] == g and gcd_table[j][k] == g:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    # 3. Precompute bad_pairs_for_x
    # bad_pairs_for_x[x] is a list of tuples (a,b) such that x is in conflict_mask[a][b]
    bad_pairs_for_x = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            mask = conflict_mask[i][j]
            # Iterate through bits in mask
            k = 1
            while k <= N:
                if mask & (1 << k):
                    bad_pairs_for_x[k].append((i, j))
                k += 1
                
    def can_add(x, S_mask):
        """Check if x can be added to set S_mask."""
        for (a, b) in bad_pairs_for_x[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True

    def greedy_add(order, S_mask=0):
        """Add elements in order to S_mask if valid."""
        S = S_mask
        for x in order:
            if not (S & (1 << x)):
                if can_add(x, S):
                    S |= (1 << x)
        return S

    def local_search(S_mask, time_limit):
        """Improve S_mask using swaps and additions."""
        S = S_mask
        best_S = S
        best_size = bin(S).count('1')
        
        while time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to add any element
            # We can pre-sort candidates by something, or just iterate
            # To speed up, we can check all x not in S
            for x in range(1, N+1):
                if not (S & (1 << x)):
                    if can_add(x, S):
                        S |= (1 << x)
                        improved = True
                        break
            
            if improved:
                if bin(S).count('1') > best_size:
                    best_S = S
                    best_size = bin(S).count('1')
                continue
            
            # Phase 2: Try swaps
            # Randomly pick a y in S and x not in S
            # To be more systematic, we can iterate
            # But random is better for escaping local optima
            # Let's try a few random swaps per iteration
            sel_list = [i for i in range(1, N+1) if S & (1 << i)]
            unsel_list = [i for i in range(1, N+1) if not (S & (1 << i))]
            
            if not unsel_list:
                break
                
            # Try up to 10 random swaps
            swap_found = False
            for _ in range(10):
                if not sel_list:
                    break
                y = random.choice(sel_list)
                x = random.choice(unsel_list)
                
                S_temp = (S & ~(1 << y)) | (1 << x)
                
                # Check if S_temp is valid
                # Since we only swapped one, we only need to check if x conflicts with pairs in S - {y}
                # And if removing y didn't break anything (it didn't, since S was valid)
                # So we just need to check if x can be added to S - {y}
                S_without_y = S & ~(1 << y)
                if can_add(x, S_without_y):
                    # S_temp is valid
                    # Try to add more elements greedily
                    S_new = S_temp
                    # Greedy add pass
                    for z in range(1, N+1):
                        if not (S_new & (1 << z)):
                            if can_add(z, S_new):
                                S_new |= (1 << z)
                    
                    new_size = bin(S_new).count('1')
                    if new_size >= best_size:
                        S = S_new
                        if new_size > best_size:
                            best_S = S
                            best_size = new_size
                        swap_found = True
                        break
            
            if not swap_found and time.time() > start_time + 5:
                # If stuck for a while, maybe restart with a different greedy seed
                # But for now, just break to outer loop if we want multiple restarts
                # Or just continue trying different random swaps
                pass

        return best_S

    # Phase 1: Initial Greedy Solutions
    best_S = 0
    best_size = 0
    
    # Strategy A: Add in order of increasing "conflict degree" (number of bad pairs)
    # Lower degree means less likely to conflict, so add them first?
    # Or higher degree means they are "harder" to place, so maybe place them later?
    # Usually, greedy independent set: pick vertex with min degree first.
    # Here, "degree" is len(bad_pairs_for_x[x]).
    
    order_by_degree = sorted(range(1, N+1), key=lambda x: len(bad_pairs_for_x[x]))
    S1 = greedy_add(order_by_degree)
    if bin(S1).count('1') > best_size:
        best_S = S1
        best_size = bin(S1).count('1')
        
    # Strategy B: Reverse order
    order_by_degree_rev = sorted(range(1, N+1), key=lambda x: -len(bad_pairs_for_x[x]))
    S2 = greedy_add(order_by_degree_rev)
    if bin(S2).count('1') > best_size:
        best_S = S2
        best_size = bin(S2).count('1')
        
    # Strategy C: Natural order
    S3 = greedy_add(range(1, N+1))
    if bin(S3).count('1') > best_size:
        best_S = S3
        best_size = bin(S3).count('1')
        
    # Strategy D: Random restarts for greedy
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        S = greedy_add(order)
        if bin(S).count('1') > best_size:
            best_S = S
            best_size = bin(S).count('1')

    # Phase 2: Local Search
    if time.time() < deadline - 5:
        best_S = local_search(best_S, deadline - 2)
        
    # Phase 3: More Local Search Restarts from different Greedy Seeds
    # If time permits, take the best few greedy solutions and run local search on them
    # But local_search already does a lot. 
    # Let's just run local_search on the current best again with a fresh timer slice?
    # No, local_search returns the best it found.
    
    # Convert bitmask to list
    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    
    if not result:
        # Fallback, though unlikely
        return [1]
        
    return result
# EVOLVE-BLOCK-END
