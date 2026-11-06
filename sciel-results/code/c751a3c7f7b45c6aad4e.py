import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    deadline = start_time + 75.0
    
    # Precompute neighbors as bitmasks
    # neighbors[i] is an integer where bit j is 1 if gcd(i+1, j+1) logic holds
    neighbors = [0] * n
    
    for i in range(n):
        num_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                mask |= (1 << j)
        neighbors[i] = mask
        
    best_size = 0
    best_set = set()
    
    # Tomita's algorithm with coloring for Max Clique
    # We use a simple greedy coloring bound
    def color_sort(vertices):
        # Returns list of vertices sorted by color, and max color
        # Simple greedy coloring
        colors = {}
        max_color = 0
        for v in vertices:
            # Find min color not used by neighbors
            used_colors = set()
            # To speed up, we can just check 1..max_color
            # Since N=512, max_color <= 512
            # But we only need to know if |R| + c <= best
            pass
        # This Python implementation of coloring might be slow.
        # Alternative: Use a simpler bound or just run Bron-Kerbosch with pivot.
        # Given N=512 and sparse-ish structure for high nodes, Bron-Kerbosch might work.
        # But worst case is exponential.
        
        # Let's try a hybrid: 
        # 1. Greedy heuristic to get a good initial bound (already done in current, but we can do it fast here).
        # 2. Branch and Bound.
        
        return []

    # Fast Greedy Initial Bound
    def greedy_max_clique():
        # Sort by degree
        degs = [bin(neighbors[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        R = set()
        P = set(range(n))
        for v in order:
            if v not in P: continue
            # Check if v is compatible with all in R
            compatible = True
            for u in R:
                if not (neighbors[v] & (1 << u)):
                    compatible = False
                    break
            if compatible:
                R.add(v)
                # Remove incompatible nodes from P
                # P = {u in P | u compatible with v}
                mask_v = neighbors[v]
                new_P = set()
                for u in P:
                    if u == v: continue
                    if mask_v & (1 << u):
                        new_P.add(u)
                P = new_P
        return R

    init_set = greedy_max_clique()
    best_size = len(init_set)
    best_set = set(init_set)
    
    # Branch and Bound
    # To avoid recursion depth issues and slow Python sets, we use integers.
    # But implementing full Tomita in Python within 60s for N=512 is risky.
    # The graph is not random.
    
    # Alternative: Use the fact that the graph is "perfect" or has structure?
    # No, it's a comparability graph of a poset? No.
    
    # Let's stick to a highly optimized Local Search with Bitsets.
    # The current program uses sets. Bitsets are faster.
    
    # Convert best_set to bitmask
    best_mask = 0
    for v in best_set:
        best_mask |= (1 << v)
        
    def get_set_from_mask(mask):
        s = set()
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            s.add(idx)
            mask ^= lsb
        return s

    # Local Search with Bitsets
    # 1-for-2 swap
    def local_search_bitset(mask, time_limit):
        current_mask = mask
        current_size = bin(current_mask).count('1')
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try 1-for-2
            # Iterate through nodes in current_mask
            m = current_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                
                # Remove u
                temp_mask = current_mask & ~(1 << u)
                temp_size = current_size - 1
                
                # Find candidates: nodes not in current_mask, compatible with all in temp_mask
                # Compatible with all in temp_mask means:
                # (neighbors[v] & temp_mask) == temp_mask
                candidates = 0
                # Iterate through all nodes not in current_mask
                # This is O(N^2) per u. Too slow?
                # N=512. N^2 = 262k.
                # Number of u is ~30.
                # 30 * 262k = 7.8M ops. Feasible in Python if optimized?
                # Bit operations are fast.
                
                # Optimization: 
                # Candidates must be in (all_nodes & ~current_mask)
                # And for each v, (neighbors[v] & temp_mask) == temp_mask
                
                # We can precompute for each v, the mask of neighbors.
                # We need v such that temp_mask is a subset of neighbors[v].
                
                # Iterate v from 0 to n-1
                for v in range(n):
                    if current_mask & (1 << v):
                        continue
                    if (neighbors[v] & temp_mask) == temp_mask:
                        candidates |= (1 << v)
                        
                if bin(candidates).count('1') < 2:
                    continue
                    
                # Find a pair in candidates that are connected
                # Iterate pairs in candidates
                c1 = candidates
                found = False
                while c1 and not found:
                    lsb1 = c1 & -c1
                    v1 = lsb1.bit_length() - 1
                    c1 ^= lsb1
                    
                    c2 = c1 & neighbors[v1]
                    if c2:
                        lsb2 = c2 & -c2
                        v2 = lsb2.bit_length() - 1
                        # Valid move
                        new_mask = temp_mask | (1 << v1) | (1 << v2)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        found = True
                        
                if improved:
                    break # Restart outer loop
            
            # If no 1-for-2, try 2-for-3?
            # Given time, maybe 1-for-2 is enough with bitsets.
            # The bitset version is much faster than set version.
            
        return current_mask, current_size

    # Run local search
    best_mask, best_size = local_search_bitset(best_mask, deadline - 5.0)
    
    # Multiple random restarts with bitset local search
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random initial set
        # Greedy with random order
        order = list(range(n))
        random.shuffle(order)
        R = 0
        for v in order:
            if (neighbors[v] & R) == R:
                R |= (1 << v)
                
        # Local search
        new_mask, new_size = local_search_bitset(R, deadline - 5.0)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
            # Continue searching from this new best?
            # local_search_bitset already ran until no improvement.
            
    # Convert back to list
    result = get_set_from_mask(best_mask)
    
    return sorted([idx + 1 for idx in result])

construct()
