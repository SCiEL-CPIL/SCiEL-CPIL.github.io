import numpy as np
def priority(adjacency):
        n = adjacency.shape[0]
        if n == 0: return np.array([], dtype=float)
        
        adj = adjacency.astype(bool)
        active = np.ones(n, dtype=bool)
        deg = adj.sum(axis=1).astype(float)
        scores = np.zeros(n)
        
        # We assign scores such that earlier picked vertices have higher scores.
        # Fixed decoder picks descending score.
        # So pick 1st gets n, 2nd gets n-1, etc.
        
        current_score = n
        active_indices = np.where(active)[0]
        
        while len(active_indices) > 0:
            # Calculate tie-breaker: sum of degrees of active neighbors
            # For each active vertex v, sum(deg[u] for u in neighbors(v) if active[u])
            
            # Efficient way:
            # neighbor_deg_sum = adj[active_indices, :].dot(deg * active)
            # But we need this for all active vertices.
            
            # Let's just compute for all active vertices.
            # deg_active = deg * active
            # neighbor_sums = adj @ deg_active
            # This gives sum of degrees of neighbors for ALL vertices.
            # We only care about active ones.
            
            neighbor_sums = adj @ (deg * active)
            
            # Filter to active vertices
            active_deg = deg[active_indices]
            active_nsum = neighbor_sums[active_indices]
            
            # Combine into a single score for sorting:
            # We want min degree, then min neighbor sum.
            # Sort key: (active_deg, active_nsum)
            # We can create a composite score for sorting.
            # Since we want min, we can just sort by these two criteria.
            
            # Create indices sorted by degree, then by neighbor sum
            # np.lexsort sorts by last key first.
            # So sort by neighbor_sum, then by degree.
            order = np.lexsort((active_nsum, active_deg))
            best_local_idx = order[0]
            best_vertex = active_indices[best_local_idx]
            
            # Assign score
            scores[best_vertex] = current_score
            current_score -= 1
            
            # Update active and degrees
            # Remove best_vertex and its neighbors
            to_remove = np.where(adj[best_vertex, :])[0]
            to_remove = np.append(to_remove, best_vertex)
            
            # Mark as inactive
            active[to_remove] = False
            
            # Update degrees of remaining active vertices
            # For each removed vertex, decrement degree of its active neighbors
            # Actually, simpler: recalculate degrees?
            # N=200, N^2 = 40000. Recalculating is O(N^2).
            # Total O(N^3) = 8,000,000. Very fast in Python/NumPy.
            
            deg = (adj * active).sum(axis=1)
            active_indices = np.where(active)[0]
            
        return scores
