import time
import random
import sys

def construct():
    start_time = time.time()
    # 150s limit, leave 35s for safety/overhead
    deadline = start_time + 115
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Limit restarts per n to avoid getting stuck on a hard n
        # but allow enough time to potentially find one
        # We descend from 40, so we spend time on 40 first.
        max_restarts_for_n = 5000
        
        found = False
        restarts = 0
        
        while time.time() < deadline - 5 and restarts < max_restarts_for_n:
            restarts += 1
            
            # 1. Generate a random maximal triangle-free graph
            # We use a randomized greedy approach:
            # Iterate pairs (u,v) in random order. Add edge if no triangle formed.
            adj = [0] * n
            all_mask = (1 << n) - 1
            
            # Generate a random permutation of all pairs
            # For n=40, pairs = 780. Shuffling 780 items is fast.
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u,v) creates a triangle
                # A triangle exists if there is a common neighbor w of u and v
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # 2. Fast Greedy Filter for Independent Set of size 10
            # If we can quickly find an independent set of size 10, discard.
            # Strategy: Try to build an independent set greedily.
            # To be more robust, we can try a few different random orderings or heuristics.
            # Here we use a randomized greedy approach: pick a random vertex, remove it and neighbors.
            
            def greedy_check_alpha_10():
                candidates = all_mask
                count = 0
                
                while candidates:
                    # Pick a random vertex from candidates
                    # Efficiently pick random bit from mask
                    r = random.getrandbits(n)
                    cand = r & candidates
                    if cand == 0:
                        # Fallback to lowest bit if random pick failed (unlikely if candidates not empty)
                        cand = candidates & -candidates
                    else:
                        # Pick a random bit from cand
                        # We can just use the lowest bit of the random intersection for speed,
                        # or iterate to find a random one. 
                        # For n=40, finding a random bit is cheap.
                        # Let's just use the lowest set bit of the intersection for simplicity and speed.
                        # This is effectively random if the random mask is uniform.
                        pass
                    
                    v_bit = cand & -cand
                    v = v_bit.bit_length() - 1
                    
                    count += 1
                    if count >= 10:
                        return True # Found independent set of size 10 (or more)
                        
                    # Remove v and its neighbors from candidates
                    # Keep only non-neighbors of v (excluding v itself)
                    # adj[v] contains neighbors. ~adj[v] contains non-neighbors and v.
                    # We want to remove v, so: candidates &= ~adj[v] & ~v_bit
                    candidates &= ~adj[v]
                    candidates &= ~v_bit
                
                return False # Did not find independent set of size 10

            if greedy_check_alpha_10():
                continue # Invalid, try next restart

            # 3. Exact Branch and Bound Verification
            # Precompute non_adj masks for efficiency
            non_adj = [0] * n
            for i in range(n):
                # Non-neighbors of i in the whole graph, excluding i itself
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # DFS to check if alpha >= 10
            # State: mask of available vertices, current size of independent set
            # Heuristic: Pick vertex with max degree in mask (to prune branches where neighbors are excluded)
            
            def has_independent_set_10():
                # Iterative or Recursive? Recursive is cleaner.
                # To avoid recursion limit issues (n=40 is fine), we use recursion.
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    if mask.bit_count() < 10 - size:
                        return False
                    
                    # Heuristic: Pick vertex with max degree in the subgraph induced by mask
                    # This helps because if we include it, we remove many vertices.
                    # If we exclude it, we keep them.
                    # Actually, for independent set, we want to pick a vertex that "blocks" many others?
                    # No, we want to find a set of non-adjacent vertices.
                    # Standard heuristic for Max Independent Set: Pick vertex with max degree.
                    
                    best_v = -1
                    max_deg = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & mask).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        temp ^= lsb
                        
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Option 1: Include best_v
                    # Next candidates are non-neighbors of best_v in mask
                    next_mask = non_adj[best_v] & mask
                    if dfs(next_mask, size + 1):
                        return True
                    
                    # Option 2: Exclude best_v
                    next_mask_excl = mask & ~v_bit
                    if dfs(next_mask_excl, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
            
        # If we exhausted restarts for this n, try next smaller n
        # We don't return here, we continue the for loop to n-1

    return best_result
