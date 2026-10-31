import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

# We will use a hybrid approach:
# 1. Base score: linear combination of distance to current and distance to destination (like inspiration 2).
# 2. Refinement: Add a "minimax" or bottleneck penalty to avoid very long edges, 
#    specifically targeting the "isolation" of remaining nodes or long jumps.
#    
# The current program (score 0.9658) uses a linear blend with a specific lambda schedule.
# Inspiration 2 (score 1.0000) uses a tuned global weight.
# Inspiration 3 (score 1.0000) uses a destination bias + isolation penalty.
#
# Strategy: 
# We will keep the linear blend structure but improve the weight schedule and add a 
# soft penalty for nodes that are "far" from the cluster of unvisited nodes (isolation proxy).
# 
# To stay within the "same family" (greedy construction with bias), we will:
# 1. Use a dynamic weight for destination bias that ramps up more aggressively at the end.
# 2. Add a small penalty based on the distance from the candidate node to the centroid 
#    of the remaining unvisited nodes. This encourages keeping the tour compact and 
#    avoiding straying too far, which acts as a smooth surrogate for minimax/isolation concerns.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1) if cost1 <= cost2 else int(c2)

    # Extract distances
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Calculate centroid of unvisited nodes for isolation/compactness penalty
    # We approximate the "compactness" by measuring distance to the mean position.
    # However, we don't have coordinates, only distance matrix.
    # A proxy for "distance to cluster" is the average distance to other unvisited nodes.
    # This is O(N^2) per call, which might be too slow for 50 nodes * 50 calls = 2500 calls * 2500 ops = 6.25M ops.
    # 6.25M ops in numpy is fast, but let's check if we can do better or if it's worth it.
    # Actually, computing avg distance to all other unvisited for each candidate is O(N^2) per candidate -> O(N^3) total.
    # N=50, N^3 = 125,000. 50 calls * 125,000 = 6.25M. This is fine.
    
    # Let's compute the average distance from each candidate to all other unvisited nodes.
    # This serves as a "connectivity" or "isolation" score. Lower is better (more connected).
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Set diagonal to 0 to exclude self, or just ignore.
    # avg_dist[i] = sum(sub_dist[i, :]) / (n_unvisited - 1)
    avg_dist = sub_dist.sum(axis=1) / (n_unvisited - 1)
    
    # Normalize scores to [0, 1] range roughly for weighting
    # dist_from_current, dist_to_dest, avg_dist
    
    # Determine destination bias weight
    # Current program: lam = 0.5 * (1.0 - n_unvisited / 50.0)
    # Let's try a stronger ramp or a different curve.
    # If n_unvisited is small, we MUST head to destination.
    # If n_unvisited is large, we should stick to local NN but keep compactness.
    
    # Let's use a weight that starts at 0.1 and goes to 0.8.
    progress = 1.0 - (n_unvisited / 50.0)
    dest_weight = 0.1 + 0.7 * progress
    
    # Connectivity weight: how much do we care about staying close to the cluster?
    # Early on, we want to explore, but not stray too far.
    # Late on, we want to close the loop, so connectivity to destination is more important.
    # Let's use a small constant weight for connectivity, e.g., 0.1.
    conn_weight = 0.1
    
    # Combine scores
    # Score = dist_from_current + dest_weight * dist_to_dest + conn_weight * avg_dist
    
    # To make weights comparable, we might want to normalize.
    # But raw distances work if the weights are tuned.
    # Let's try:
    scores = dist_from_current + dest_weight * dist_to_dest + conn_weight * avg_dist
    
    # Find min
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
