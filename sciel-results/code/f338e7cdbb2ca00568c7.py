"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a dynamic geometric heuristic.
    
    Strategy:
    1. Calculate the centroid of the remaining unvisited nodes.
    2. Determine the 'progress' of the tour (how many nodes are left).
    3. Blend the objective:
       - Early/Mid tour: Minimize distance to the centroid of remaining nodes 
         (encourages staying in the middle of the cluster, avoiding dead ends).
       - Late tour: Minimize distance to the destination (start node) 
         (ensures a cheap return leg).
    
    This avoids the "nearest neighbor" trap of forming a tight cluster far from 
    the rest of the points or the start, and avoids the "pure destination" trap 
    of rushing home too early.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_total = len(distance_matrix)
    n_remaining = len(unvisited_nodes)
    
    # Progress: 0.0 at start (many nodes left), 1.0 near end (few nodes left)
    # We want the "return home" bias to kick in when we have a small fraction left.
    # Let's define a threshold. If > 50% visited, start worrying about return.
    # Actually, let's make it smooth.
    
    # Calculate centroid of unvisited nodes
    # distance_matrix is NxN. We don't have coordinates, but we can approximate 
    # "distance to centroid" by the average distance to all unvisited nodes?
    # No, that's O(N^2) per step if not careful, but N=50 is small.
    # However, distance_matrix is already computed. 
    # "Distance to centroid" in Euclidean space is not simply the average distance.
    # But minimizing the average distance to remaining nodes is a good proxy for 
    # "staying central" to the remaining cluster.
    
    # Cost 1: Distance from current to candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Cost 2: "Centrality" - average distance from candidate to all other unvisited nodes
    # This encourages picking a node that is close to the rest of the unvisited set,
    # preventing the tour from stretching out to a distant outlier if a closer 
    # central option exists.
    # Note: distance_matrix[candidate, unvisited_nodes] includes distance to itself (0).
    # We want the mean distance to the *other* unvisited nodes.
    
    # To avoid O(N^2) in the inner loop of the TSP construction (which is already O(N^2) 
    # overall), we can compute this efficiently.
    # Actually, for N=50, even a simple loop or matrix operation is fast enough.
    # Let's compute the average distance from each candidate to the set of unvisited nodes.
    
    # dists_to_unvisited is a matrix of shape (n_remaining, n_remaining)
    # distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Sum of distances to all unvisited (including self, which is 0)
    sums = np.sum(sub_matrix, axis=1)
    # Average distance to other unvisited nodes (approx: sum / (n_remaining - 1))
    # Using n_remaining is fine as a scaling factor, it's a constant for this step.
    avg_dist_to_unvisited = sums / (n_remaining - 1)
    
    # Cost 3: Distance to destination (start node)
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Dynamic weighting
    # If n_remaining is large, we care more about centrality (avg_dist_to_unvisited).
    # If n_remaining is small, we care more about returning to dest.
    
    # Let's define alpha (weight for centrality) and beta (weight for destination).
    # alpha + beta = 1.0 for simplicity in scaling, or just relative weights.
    
    # Transition function:
    # When n_remaining > n_total * 0.5, we are in "expansion" mode.
    # When n_remaining < n_total * 0.5, we transition to "return" mode.
    
    # A smooth transition:
    # fraction_remaining = n_remaining / n_total
    # weight_dest = 1.0 - fraction_remaining  (0 at start, 1 at end)
    # weight_central = fraction_remaining     (1 at start, 0 at end)
    
    # However, "centrality" (avg dist to unvisited) is a good proxy for "short edges" 
    # in the middle of the tour. "Distance to dest" is crucial for the final leg.
    # "Distance from current" is always crucial.
    
    # Let's try:
    # Score = dists_from_current + w_central * avg_dist_to_unvisited + w_dest * dists_to_dest
    
    # w_central should be high early, w_dest high late.
    # Let's use a sigmoid or linear ramp.
    # Linear ramp based on fraction remaining.
    frac_rem = n_remaining / n_total
    
    # If frac_rem is 1.0 (start), w_central = 1.0, w_dest = 0.0
    # If frac_rem is 0.0 (end), w_central = 0.0, w_dest = 1.0
    # But we never reach 0.0 (we stop at 1).
    
    # To make the transition sharper, we can use a power function or piecewise.
    # Let's try: w_dest = (1 - frac_rem) ** 2 ? No, that's too slow.
    # Let's try: w_dest = 1 - frac_rem
    # w_central = frac_rem
    
    # Is "avg_dist_to_unvisited" the right metric for centrality?
    # It penalizes nodes that are far from the *remaining* cluster.
    # This is effectively a "nearest neighbor in the remaining set" heuristic, 
    # but averaged. It's a form of "minimum spanning tree" intuition.
    
    # Let's refine the weights.
    # Early on, we want to build a compact tour. 
    # Late on, we want to close the loop.
    
    # Let's add a small constant bias to ensure dists_from_current is dominant 
    # to avoid weird jumps, but the other terms guide the choice.
    
    w_central = frac_rem
    w_dest = 1.0 - frac_rem
    
    # Normalize the terms? They are all distances, so same scale.
    # But avg_dist_to_unvisited is an average, so it might be smaller than a direct dist.
    # Let's scale avg_dist_to_unvisited by a factor if needed, but for now, 
    # let's trust the relative magnitude.
    
    # Actually, avg_dist_to_unvisited is typically smaller than dists_from_current 
    # if the tour is expanding. 
    # Let's boost the centrality term slightly to make it competitive.
    # Or, let's use the *minimum* distance to unvisited (nearest neighbor in remaining) 
    # instead of average? 
    # Min distance is "greedy on remaining set". 
    # Average distance is "centroid-like".
    # Let's stick with average for "compactness".
    
    scores = dists_from_current + w_central * avg_dist_to_unvisited + w_dest * dists_to_dest
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
