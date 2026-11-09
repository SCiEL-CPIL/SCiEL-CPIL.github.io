import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    active = np.ones(n, dtype=bool)
    scores = np.zeros(n, dtype=float)
    
    for step in range(n):
        if not np.any(active):
            break
        
        # Calculate degrees in the active subgraph
        deg = (adjacency @ active).astype(int)
        
        # Calculate sum of neighbor degrees
        sum_nei_deg = (adjacency @ deg).astype(int)
        
        # Heuristic: Minimum Degree, tie-break by Minimum Sum of Neighbor Degrees
        # We want to minimize deg, then sum_nei_deg.
        # To do this, we can find the min deg, then among those, min sum_nei_deg.
        
        min_deg = np.min(deg[active])
        # Candidates are active nodes with min_deg
        candidates = active & (deg == min_deg)
        
        # If no candidates (should not happen), fallback
        if not np.any(candidates):
            break
            
        # Among candidates, pick the one with min sum_nei_deg
        # Mask non-candidates with high value
        cost = sum_nei_deg.copy()
        cost[~candidates] = n + 1
        
        v = np.argmin(cost)
        
        # Assign score
        scores[v] = n - step
        
        # Remove v and neighbors
        active[v] = False
        active[adjacency[v]] = False
        
    return scores
