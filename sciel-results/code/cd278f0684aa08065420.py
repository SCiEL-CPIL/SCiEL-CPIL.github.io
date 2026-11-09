# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    degree = A.sum(axis=1).astype(np.float64)
    
    # Precompute neighbor lists
    neighbors = [np.where(A[i])[0] for i in range(n)]
    
    # Core idea: For each vertex v, compute its "effective territory size"
    # This simulates: if v were selected first in the greedy process, 
    # how many vertices would remain that are NOT neighbors of v?
    # But we go deeper: we simulate the full greedy process starting with v,
    # and count how many vertices get selected.
    # 
    # This is expensive, so we use an approximation:
    # territory(v) = number of vertices u such that:
    #   - u is not a neighbor of v
    #   - u has "low contention" with other vertices that are also not neighbors of v
    #
    # A cheaper proxy: territory(v) = (n - 1 - degree[v]) * (1 - local_density(v))
    # where local_density(v) measures how connected v's non-neighborhood is.
    #
    # Even better: use BFS to find the size of the connected component of v
    # in the graph where we remove all neighbors of v. This gives the "uncontested region"
    # that v can potentially claim.
    
    scores = np.zeros(n, dtype=np.float64)
    
    # First pass: quick degree-based score as baseline
    for v in range(n):
        if time_left() < 10.0:
            # Fallback: simple degree heuristic
            return -degree.astype(np.float64)
        
        # Remove v and its neighbors, find connected component containing... 
        # Actually, let's think differently.
        
        # The key insight from the directive: model greedy selection as influence propagation.
        # When v is selected, it "blocks" its neighbors. The remaining graph has 
        # n - 1 - degree[v] vertices. The quality of v's selection depends on how 
        # well the remaining vertices can form an independent set.
        
        # Approximation: the remaining graph's independence number is roughly 
        # proportional to (n - 1 - degree[v]) / (1 + avg_degree_remaining)
        
        # But we can do better: compute the actual connected component structure.
        
        # For vertex v, create a modified adjacency where v and its neighbors are removed.
        # Then find the largest connected component in the remaining graph.
        # The size of this component indicates how much "territory" v has left to work with.
        
        # However, this is O(n^2) per vertex, which is O(n^3) total = 8M for n=200.
        # That's feasible within time limits.
        
        # Build mask of vertices NOT blocked by v
        blocked = np.zeros(n, dtype=bool)
        blocked[v] = True
        if len(neighbors[v]) > 0:
            blocked[neighbors[v]] = True
        
        # Find connected components in the unblocked subgraph
        # Use BFS from each unblocked vertex to find component sizes
        visited = np.zeros(n, dtype=bool)
        max_component = 0
        
        for u in range(n):
            if not blocked[u] and not visited[u]:
                # BFS from u
                queue = deque([u])
                visited[u] = True
                comp_size = 0
                while queue:
                    w = queue.popleft()
                    comp_size += 1
                    for x in neighbors[w]:
                        if not blocked[x] and not visited[x]:
                            visited[x] = True
                            queue.append(x)
                max_component = max(max_component, comp_size)
        
        # Score: prefer vertices that leave a large, well-connected territory
        # Low degree is good (blocks fewer vertices), large remaining component is good
        # But we also want to penalize if the remaining component is highly connected
        # (high density means harder to find independent set)
        
        # Compute density of the largest component
        # Actually, let's just use the component size as a proxy for "available territory"
        
        # Normalize: territory_size / (n - 1 - degree[v]) gives fraction of available vertices
        # that are in the largest component
        available = n - 1 - int(degree[v])
        if available > 0:
            territory_fraction = max_component / available
        else:
            territory_fraction = 0.0
        
        # Final score: combine degree (lower is better) and territory quality
        # score = -degree + territory_fraction * scale
        # The territory_fraction should dominate for vertices with similar degrees
        
        scores[v] = -degree[v] + 50.0 * territory_fraction
    
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
