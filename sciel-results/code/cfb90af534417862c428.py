import time
import random

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a 135s deadline to be safe.
    deadline = start_time + 135
    
    # Fallback: Andrásfai graph And(9) on 26 vertices
    # It is triangle-free and has independence number 9.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    
    # Helper to extract edges from adjacency bitmasks
    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    # Helper to check if the graph has an independent set of size 10
    # If it does NOT, the graph is valid (alpha < 10).
    def has_independent_set_10(adj, n):
        all_mask = (1 << n) - 1
        # Precompute non-neighbor masks for faster lookups in DFS
        # non_neighbors[i] contains bits for j != i such that (i,j) is NOT an edge
        non_neighbors = [0] * n
        for i in range(n):
            # all_except_i is all bits set except i
            # ~adj[i] has 0s where edges exist, 1s where they don't (and padding)
            # We mask with all_except_i to remove i itself and padding
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
        # Recursive DFS with pruning
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            # Pruning: if remaining candidates < needed, impossible
            if mask.bit_count() < 10 - size:
                return False
            
            # Heuristic: Pick vertex v in mask with minimum number of non-neighbors in mask.
            # This vertex is the "most constrained" in terms of exclusion if we include it,
            # or rather, including it removes the fewest options? 
            # Actually, standard heuristic for finding independent set:
            # If we include v, we must exclude all its neighbors.
            # The new mask becomes mask & ~neighbors[v].
            # To maximize pruning, we want to choose v such that the new mask is smallest.
            # Equivalent to choosing v with maximum degree in the induced subgraph?
            # Or minimum non-neighbors?
            # Let's stick to the proven heuristic from Inspiration 1:
            # Pick v with minimum popcount(non_neighbors[v] & mask).
            # Why? If non_neighbors[v] is small, then including v leaves a small set.
            # This helps prove non-existence quickly?
            # Actually, for finding a set, we usually want to branch on the most constrained.
            
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                # Count how many vertices in 'mask' are non-neighbors of v
                # This is the size of the set we would keep if we picked v
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            # The new available vertices are those in 'mask' that are non-neighbors of best_v
            # Note: non_neighbors[best_v] already excludes best_v itself
            new_mask_include = non_neighbors[best_v] & mask
            
            if dfs(new_mask_include, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                return True
                
            return False
            
        return dfs(all_mask, 0)

    # Main search loop
    # We try to find a graph with n vertices, starting from the maximum possible (40)
    # down to 27 (since we have a seed for 26).
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Number of restarts is dynamic based on time
        restarts = 0
        while time.time() < deadline - 2:  # Keep 2s buffer for final return
            restarts += 1
            if restarts > 10000: # Hard cap to prevent infinite loop if check is super fast
                break
                
            # Generate a random maximal triangle-free graph
            adj = [0] * n
            
            # Generate all pairs and shuffle
            # For n=40, there are 780 pairs. This is cheap.
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u, v) creates a triangle
                # A triangle exists if u and v have a common neighbor
                common = adj[u] & adj[v]
                if common == 0:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            # If has_independent_set_10 returns False, then alpha < 10, so graph is valid.
            if not has_independent_set_10(adj, n):
                edges = edges_from_adj(adj, n)
                return {'n': n, 'edges': edges}
        
    return best_result
