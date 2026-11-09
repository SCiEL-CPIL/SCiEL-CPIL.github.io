import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    t0 = time.time()
    deadline = t0 + 280.0
    
    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()

    result = edge_distance.copy()
    
    # Calculate median distance for scaling
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0

    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0

    # Iterate through edges in the current tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Find neighbors in the tour
        # u is connected to p and v
        # v is connected to u and n
        p = local_opt_tour[(i - 1) % n]
        n_next = local_opt_tour[(i + 2) % n]
        
        # Calculate potential 2-opt improvement for removing (u, v)
        # Current segment: p-u-v-n
        # Cost: d(p,u) + d(u,v) + d(v,n)
        # New segment: p-v-u-n (or p-u-n-v? No, 2-opt reverses the segment between u and v? 
        # Standard 2-opt on edge (u,v) in a cycle:
        # Remove (u,v) and (p, n_next)? No.
        # 2-opt move on edge (u,v) where u is at i and v is at i+1:
        # The edge (u,v) is part of the tour.
        # Neighbors of u are p and v.
        # Neighbors of v are u and n_next.
        # If we remove (u,v), we break the cycle into a path p...u and v...n_next.
        # To close it, we connect p to n_next? No, that would be a different move.
        # The standard 2-opt move that removes edge (u,v) involves the edge (p, n_next) as well?
        # Actually, 2-opt considers pairs of edges.
        # If we want to specifically penalize (u,v) to make 2-opt remove it, 
        # we should look at the best 2-opt move involving (u,v).
        # The move is: replace (p,u) and (v,n_next) with (p,v) and (u,n_next)? 
        # No, that's not a valid 2-opt on a cycle unless we reverse the segment.
        # In a cycle, removing (u,v) and (p, n_next) is not a single 2-opt step unless 
        # p and n_next are adjacent? No.
        
        # Let's look at the standard 2-opt improvement calculation for an edge (u,v) in the tour.
        # The tour is ... p, u, v, n ...
        # A 2-opt move that removes (u,v) would replace (p,u) and (v,n) with (p,v) and (u,n) ??
        # No, that creates a crossing if the tour is planar, but generally:
        # The move is to connect p to v and u to n.
        # This is valid if we reverse the segment between u and v? 
        # If the tour is p -> u -> v -> n, and we change to p -> v -> u -> n, 
        # we removed (p,u) and (v,n) and added (p,v) and (u,n).
        # But (u,v) is still there? No, the segment u->v is reversed to v->u. 
        # The edge (u,v) is effectively removed and replaced by (v,u) which is the same edge.
        # Wait, 2-opt on a cycle removes two edges and adds two.
        # If we want to remove (u,v), we must remove another edge (a,b) and add (u,a) and (v,b) or similar.
        
        # Actually, the most direct way to see if (u,v) is "bad" is to check if 
        # d(p, v) + d(u, n_next) < d(p, u) + d(u, v) + d(v, n_next) - d(p, n_next)? 
        # No.
        
        # Let's simplify. 
        # A common heuristic is to penalize edges that are "long" relative to their neighbors.
        # Or just stick to usage-based penalties which are proven to work in GLS.
        # The "edge-reversal" hint might refer to a specific implementation detail.
        
        # Let's try a robust usage-based penalty with a slight boost for "long" edges in the tour.
        # Long edges in a local optimum are often bad.
        
        used = edge_n_used[u, v]
        
        # Base penalty scaled by median distance
        # We want to encourage 2-opt to remove this edge.
        # A penalty of ~0.1-0.5 * med_dist is typical.
        
        # Usage scaling
        usage_factor = 1.0 + 0.1 * used
        
        # Edge length penalty (optional)
        # If the edge is much longer than the median, penalize more.
        edge_len = edge_distance[u, v]
        len_factor = 1.0 + 0.5 * max(0, (edge_len - med_dist) / med_dist)
        
        penalty = 0.2 * med_dist * usage_factor * len_factor
        
        result[u, v] += penalty
        result[v, u] += penalty
        
    return result
