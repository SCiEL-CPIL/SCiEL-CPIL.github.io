import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for evaluation overhead
    
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        # The number of restarts can be dynamic based on time remaining
        restarts = 0
        while time.time() < deadline - 5:  # Keep 5s buffer
            restarts += 1
            if restarts > 5000: # Cap restarts to prevent infinite loop if check is very fast
                break
                
            # Generate a random maximal triangle-free graph
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            
            # Generate all pairs and shuffle
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is a common neighbor
                common = adj[u] & adj[v]
                if common == 0:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            # We use a branch and bound to see if an independent set of size 10 exists.
            # If it does NOT exist, the graph is valid.
            
            # Precompute non-neighbor masks
            # non_neighbors[i] contains bits for j != i such that not (adj[i] & (1<<j))
            non_neighbors = [0] * n
            for i in range(n):
                # All vertices except i
                all_except_i = (1 << n) - 1 - (1 << i)
                # Non-neighbors are those in all_except_i that are NOT in adj[i]
                non_neighbors[i] = all_except_i & ~adj[i]
            
            # Branch and bound for independent set of size 10
            # We want to know if there is an independent set of size 10.
            # If we return False, it means no such set exists, so alpha < 10.
            
            def has_independent_set_10():
                # State: mask of available vertices, current size of independent set
                # We use recursion with memoization or just simple DFS with pruning
                # Since n is small (<=40), and we prune aggressively, this should be fast.
                
                # To speed up, we can use a greedy upper bound:
                # If popcount(mask) < 10 - current_size, return False (can't reach size 10)
                # Actually, we want to find if size 10 is possible.
                
                # Let's implement a recursive search that tries to build a set of size 10.
                # If it succeeds, return True. If it exhausts possibilities, return False.
                
                # Optimization: Order vertices by degree in the induced subgraph?
                # Or just pick the one with fewest non-neighbors in the current mask?
                # Fewer non-neighbors means including it prunes the search space more.
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    # Pruning: if number of bits in mask < 10 - size, impossible
                    if bin(mask).count('1') < 10 - size:
                        return False
                    
                    # Pick a vertex from mask
                    # Heuristic: pick vertex v in mask with minimum popcount(non_neighbors[v] & mask)
                    # This is the "most constrained" vertex.
                    # Finding this takes O(n) per step.
                    
                    # For speed, we might just pick the lowest set bit if n is small.
                    # But the heuristic is important for proving non-existence.
                    
                    # Let's try the heuristic if mask has > 15 bits, otherwise pick first.
                    if bin(mask).count('1') > 15:
                        min_count = n + 1
                        best_v = -1
                        temp_mask = mask
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            v = lsb.bit_length() - 1
                            # Count non-neighbors of v in mask
                            cnt = bin(non_neighbors[v] & mask).count('1')
                            if cnt < min_count:
                                min_count = cnt
                                best_v = v
                            temp_mask ^= lsb
                    else:
                        # Just pick the lowest set bit
                        lsb = mask & -mask
                        best_v = lsb.bit_length() - 1
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Option 1: Include best_v
                    # New mask: non-neighbors of best_v within current mask, excluding best_v itself
                    # non_neighbors[best_v] already excludes best_v? 
                    # My definition: all_except_i & ~adj[i]. So it excludes i.
                    # So new_mask = non_neighbors[best_v] & mask
                    new_mask_include = non_neighbors[best_v] & mask
                    
                    if dfs(new_mask_include, size + 1):
                        return True
                    
                    # Option 2: Exclude best_v
                    new_mask_exclude = mask & ~v_bit
                    if dfs(new_mask_exclude, size):
                        return True
                    
                    return False
                
                return dfs((1 << n) - 1, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    # Only add edges where i < j to avoid duplicates
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    return best_result
