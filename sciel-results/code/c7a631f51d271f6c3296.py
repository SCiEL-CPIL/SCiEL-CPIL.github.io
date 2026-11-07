import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using a surgical tour-edge inflation strategy.
    
    Strategy:
    1. Surgical Perturbation: Instead of penalizing a broad set of edges based on usage 
       history and distance (which distorts the global geometry), we only modify the 
       edges that form the current local-optimal tour.
    2. Preserved Geometry: All edges NOT in the current tour retain their true Euclidean 
       distances. This allows 2-opt to explore alternative connections without being 
       misled by artificial penalties on non-tour edges.
    3. Adaptive Inflation: The inflation factor for tour edges is scaled by the 
       cumulative usage count of those specific edges. Edges that have been used more 
       frequently in past iterations receive a higher penalty, encouraging the search 
       to break them more aggressively.
    4. Bounded Impact: The multiplier is capped to prevent extreme distortions that 
       might push 2-opt into poor local optima.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Ensure we have a valid tour
    if local_opt_tour is None or len(local_opt_tour) < 2:
        return d_out

    tour = np.array(local_opt_tour)
    
    # We will identify all edges in the current tour and apply a penalty multiplier.
    # The penalty is multiplicative: d_new = d_old * multiplier
    
    # Determine the multiplier for each tour edge.
    # Base multiplier: 1.0 (no change)
    # We want to increase the cost of tour edges to force 2-opt to remove them.
    
    # Strategy for multiplier:
    # Let usage_i be the count of edge i in edge_n_used.
    # Multiplier_i = 1 + strength * (usage_i / max_usage) * (1 / d_i) * median_d
    # This makes short, frequently used edges very expensive.
    # However, the directive says: "apply a high penalty factor only to edges in the 
    # current local-optimal tour, while leaving all other edges at their true distances".
    
    # Let's use a simpler, robust multiplier based on usage.
    # If an edge is used many times, it's likely a "bad" edge stuck in the local opt.
    # We'll use a multiplier that grows with usage count.
    
    # Get usage counts for all edges in the tour
    tour_edges = []
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tour_edges.append((a, b))
    # Close the tour
    a, b = tour[-1], tour[0]
    tour_edges.append((a, b))
    
    # Calculate a global scale factor for usage to normalize penalties
    # We look at the max usage count in the entire matrix to set a reference
    max_usage_global = np.max(edge_n_used) if edge_n_used.size > 0 else 1
    if max_usage_global == 0:
        max_usage_global = 1.0
        
    # For each tour edge, calculate a multiplier
    # We want the multiplier to be > 1.
    # Let's use: multiplier = 1 + alpha * (usage_edge / max_usage_global)
    # But we also want to consider that shorter edges are more "valuable" in TSP, 
    # so breaking a short edge is more disruptive. 
    # The prompt suggests: "high penalty factor only to edges in the current local-optimal tour".
    
    # Let's try a fixed high penalty for all tour edges, but scaled slightly by usage.
    # If we just multiply all tour edges by 2.0, 2-opt will likely remove the longest 
    # tour edges first if they are part of a 2-opt swap that improves the tour.
    # But we want to break the tour to escape the local optimum.
    
    # Let's use a multiplier that is significant, e.g., 2.0 to 5.0.
    # To make it adaptive:
    # multiplier = 1 + 2.0 * (usage_edge / max_usage_global)
    # If usage is high (close to max), multiplier is ~3.
    # If usage is low, multiplier is ~1.
    
    # However, early in the search, usage counts are low. We might want a base penalty.
    # Let's use: multiplier = 1 + base_penalty + adaptive_penalty
    # base_penalty = 1.0 (so min multiplier is 2.0)
    # adaptive_penalty = 2.0 * (usage_edge / max_usage_global)
    
    base_penalty = 1.0
    adaptive_strength = 2.0
    
    for a, b in tour_edges:
        usage = edge_n_used[a, b]
        # Normalize usage to [0, 1]
        norm_usage = usage / max_usage_global
        multiplier = 1.0 + base_penalty + adaptive_strength * norm_usage
        
        # Apply to both directions
        d_out[a, b] *= multiplier
        d_out[b, a] *= multiplier
        
    # Ensure symmetry (should already be symmetric if input was, but just in case)
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure finite and non-negative
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return d_out
        
    return d_out
