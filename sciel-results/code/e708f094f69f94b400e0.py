import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty using a repulsive force model.
    
    Strategy:
    1. Identify edges in the current local optimum tour.
    2. Calculate a repulsive penalty for each edge in the tour.
    3. The penalty magnitude is inversely proportional to the edge's true distance (1/d).
       This means short edges (which are typically "good" and kept in local optima) 
       receive larger penalties relative to their length, forcing the 2-opt search 
       to consider swapping them out for longer, less-used edges.
    4. The penalty is scaled so that the total added cost to the tour is approximately 
       a fixed percentage (e.g., 10%) of the current tour length, ensuring the 
       perturbation is significant but not destructive.
    5. Edges not in the current tour receive no penalty (or a very small one if 
       recently used, but here we focus on the current tour's edges to break the 
       current local optimum).
    
    Returns a perturbed distance matrix.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Get edges in the current local optimum tour
    tour = np.array(local_opt_tour)
    if len(tour) < 2:
        return d_out
    
    # Identify edges in the current tour
    # We only penalize edges that are currently in the tour to break the local optimum.
    # This is the core of Guided Local Search: penalize the decisions that led to the local optimum.
    
    tour_edges = []
    for i in range(len(tour) - 1):
        tour_edges.append((tour[i], tour[i + 1]))
    # Close the tour
    tour_edges.append((tour[-1], tour[0]))
    
    # Calculate current tour length for scaling
    tour_length = 0.0
    for a, b in tour_edges:
        tour_length += edge_distance[a, b]
        
    if tour_length <= 0:
        return d_out
        
    # Target perturbation: We want the sum of penalties on tour edges to be ~10% of tour length.
    # This is a heuristic choice. Too small -> no escape. Too large -> random search.
    target_penalty_total = 0.10 * tour_length
    
    # Calculate raw repulsive weights for tour edges.
    # Weight is inversely proportional to distance.
    # To avoid division by zero (though unlikely in uniform random points), add epsilon.
    eps = 1e-9
    weights = []
    for a, b in tour_edges:
        d = edge_distance[a, b]
        if d < eps:
            w = 1.0
        else:
            w = 1.0 / d
        weights.append(w)
        
    weights = np.array(weights)
    
    # Normalize weights so that sum(weights) = 1
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        # Fallback: uniform penalty if something went wrong
        weights = np.ones(len(tour_edges)) / len(tour_edges)
        
    # Apply penalties
    # Penalty for edge (a,b) = target_penalty_total * weight_i
    for i, (a, b) in enumerate(tour_edges):
        penalty = target_penalty_total * weights[i]
        d_out[a, b] += penalty
        d_out[b, a] += penalty
        
    # Ensure symmetry (just in case)
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure finite values
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    
    # Ensure non-negative distances
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return d_out
        
    return d_out
