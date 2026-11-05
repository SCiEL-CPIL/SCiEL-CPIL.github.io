"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin for safety
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad triples and conflicts
    # conflicts[x] will be a list of pairs (a, b) such that (a, b, x) is a bad triple.
    # We use 0-indexed for bitmasks? No, 1-indexed is fine if we shift by (val-1).
    conflicts = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # (a, b, c) is a bad triple
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # Sort conflicts lists to maybe help cache locality? Not strictly necessary.
    # But we can also remove duplicates if any (though logic above ensures unique triples).
    
    def is_compatible(x, mask_S):
        # mask_S is an integer bitset where bit i-1 is set if i is in S
        for a, b in conflicts[x]:
            # Check if both a and b are in mask_S
            if ((mask_S >> (a - 1)) & 1) and ((mask_S >> (b - 1)) & 1):
                return False
        return True

    def greedy_construct(order):
        mask_S = 0
        S_list = []
        for x in order:
            if is_compatible(x, mask_S):
                mask_S |= (1 << (x - 1))
                S_list.append(x)
        return S_list, mask_S

    def local_search(S_list, mask_S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            U_list = [x for x in range(1, N + 1) if not ((mask_S >> (x - 1)) & 1)]
            
            # 1. Try to add any element
            for v in U_list:
                if is_compatible(v, mask_S):
                    mask_S |= (1 << (v - 1))
                    S_list.append(v)
                    improved = True
                    # Update U_list? It's small enough to rebuild or just break and let next loop handle.
                    # Breaking to restart the loop with new state is safer and cleaner.
                    break
            
            if improved:
                continue

            # 2. Try swaps
            found_swap = False
            # We need to be careful with indices if we modify S_list.
            # We will iterate through a copy of indices.
            for i in range(len(S_list)):
                u = S_list[i]
                # Temporarily remove u
                mask_S_minus_u = mask_S & ~(1 << (u - 1))
                
                for v in U_list:
                    if is_compatible(v, mask_S_minus_u):
                        # Swap u with v
                        S_list[i] = v
                        mask_S = mask_S_minus_u | (1 << (v - 1))
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
                    
        return S_list, mask_S

    best_size = 0
    best_set = []
    
    # Strategy:
    # 1. Greedy by degree (fewest conflicts first)
    # 2. Local Search
    # 3. Randomized restarts with perturbations
    
    # Pre-compute degree for greedy ordering
    degrees = [len(conflicts[x]) for x in range(N + 1)]
    
    # Initial Greedy: Sort by degree ascending
    order_by_degree = sorted(numbers, key=lambda x: degrees[x])
    
    S_list, mask_S = greedy_construct(order_by_degree)
    best_set = S_list[:]
    best_size = len(S_list)
    
    # Local Search on the best found so far
    S_list, mask_S = local_search(S_list, mask_S)
    if len(S_list) > best_size:
        best_size = len(S_list)
        best_set = S_list[:]
        
    # Randomized Restarts
    # We use a few restarts with random orders or perturbed orders
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random shuffle
        random_order = numbers[:]
        random.shuffle(random_order)
        
        S_list, mask_S = greedy_construct(random_order)
        
        # Local Search
        S_list, mask_S = local_search(S_list, mask_S)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
    # Final verification (optional but good for safety)
    # Convert best_set to set for quick check? 
    # The mask ensures validity during construction, so it should be valid.
    
    return best_set

# EVOLVE-BLOCK-END
