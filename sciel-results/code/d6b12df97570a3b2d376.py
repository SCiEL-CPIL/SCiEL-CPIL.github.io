import random
import time

def construct():
    """
    Constructs a triangle-free graph on n vertices with independence number <= 9.
    Uses simulated annealing to find a valid graph on n=40.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Target n
    n = 40
    
    # Initialize with a random graph
    # Start with a sparse graph to have fewer triangles initially
    p = 0.3
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Helper to count triangles
    def count_triangles(adj):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            count += 1
        return count

    # Helper to find a large independent set (greedy)
    def greedy_independent_set(adj):
        remaining = list(range(n))
        indep_set = []
        while remaining:
            # Pick vertex with minimum degree in induced subgraph
            min_deg = float('inf')
            min_idx = -1
            for idx, v in enumerate(remaining):
                deg = sum(adj[v][u] for u in remaining if u != v)
                if deg < min_deg:
                    min_deg = deg
                    min_idx = idx
            v = remaining.pop(min_idx)
            indep_set.append(v)
            # Remove neighbors
            remaining = [u for u in remaining if not adj[v][u]]
        return len(indep_set)

    # Helper to check if alpha <= 9 exactly (branch and bound)
    def has_independent_set_of_size_k(adj, k):
        if k <= 0:
            return False
        if k == 1:
            return True
        
        candidates = list(range(n))
        
        def dfs(current_set, candidates):
            if len(current_set) >= k:
                return True
            if not candidates:
                return False
            # Pruning: if remaining candidates + current set size < k, fail
            if len(current_set) + len(candidates) < k:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with fewest neighbors in candidates
            v = candidates[0]
            
            # Option 1: Include v
            new_candidates = [u for u in candidates if u != v and not adj[v][u]]
            if dfs(current_set + [v], new_candidates):
                return True
            
            # Option 2: Exclude v
            if dfs(current_set, candidates[1:]):
                return True
                
            return False

        return dfs([], candidates)

    # Initial cost
    triangles = count_triangles(adj)
    alpha_greedy = greedy_independent_set(adj)
    cost = triangles + 100 * max(0, alpha_greedy - 9)
    
    best_adj = [row[:] for row in adj]
    best_cost = cost
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Flip edge
        adj[i][j] ^= 1
        adj[j][i] ^= 1
        
        # Calculate new cost
        new_triangles = count_triangles(adj)
        new_alpha_greedy = greedy_independent_set(adj)
        new_cost = new_triangles + 100 * max(0, new_alpha_greedy - 9)
        
        delta = new_cost - cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_adj = [row[:] for row in adj]
                # If we found a valid graph (cost 0), we can verify and return
                if best_cost == 0:
                    # Verify alpha exactly
                    if not has_independent_set_of_size_k(best_adj, 10):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if best_adj[u][v]:
                                    edges.append([u, v])
                        return {'n': n, 'edges': edges}
        else:
            # Revert
            adj[i][j] ^= 1
            adj[j][i] ^= 1
            
        T *= cooling
        
    # If we didn't find a perfect graph, try to return the best one if it's valid
    # Or just return the best found, but it might be invalid.
    # The task requires a VALID graph. If we don't find one, we should fall back to a known valid one.
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    n_fallback = 26
    edges_fallback = [[i, j] for i in range(n_fallback) for j in range(i + 1, n_fallback) if (j - i) % n_fallback % 3 == 1]
    return {'n': n_fallback, 'edges': edges_fallback}

# EVOLVE-BLOCK-END
