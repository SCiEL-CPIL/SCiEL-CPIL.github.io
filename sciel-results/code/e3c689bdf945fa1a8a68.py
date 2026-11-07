"""Guided local search penalty rule for TSP with adaptive kick and edge-specific penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for tracking rounds and timing
_round_counter = 0
_start_time = None
_DEADLINE_FRACTION = 0.9  # Use 90% of the 330s budget


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Track the round number to adapt penalty strength.
    2. Apply a kick to edges in the current local optimum tour.
    3. The kick magnitude increases with usage count to encourage diversification.
    4. Use a smooth, bounded penalty to avoid distorting the metric too much.
    """
    global _round_counter, _start_time
    
    # Initialize timing if not already done
    if _start_time is None:
        _start_time = time.monotonic()
    
    # Safety check: if we're running out of time, return the original distances
    elapsed = time.monotonic() - _start_time
    if elapsed > 330.0 * _DEADLINE_FRACTION:
        return edge_distance.copy()
    
    # Increment round counter
    _round_counter += 1
    
    n = edge_distance.shape[0]
    
    # Extract edges from the current tour
    # local_opt_tour is a permutation of [0, 1, ..., n-1]
    # Edges are (tour[i], tour[(i+1) % n])
    tour = local_opt_tour
    edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        edges.append((u, v))
    
    # Create the modified distance matrix
    modified_dist = edge_distance.copy()
    
    # Calculate adaptive kick magnitude
    # Start with a moderate kick and increase it over time
    # The kick is applied to edges in the current tour
    # We use the usage count to scale the penalty
    
    # Get the maximum usage count to normalize
    max_used = edge_n_used.max()
    if max_used == 0:
        max_used = 1
    
    # Adaptive kick: increases with round number but saturates
    # Base kick is proportional to the edge's usage count
    # We use a smooth function to avoid discontinuities
    
    # For each edge in the current tour, apply a penalty
    for u, v in edges:
        # Usage count for this edge
        count = edge_n_used[u, v]
        
        # Normalize usage: count / max_used gives relative frequency
        # But we also want to consider the absolute count
        # Use a combination: penalty grows with count but is bounded
        
        # Kick magnitude: starts small, grows with usage
        # We use a tanh-like saturation to keep it bounded
        # Base kick is around 0.5 * edge_distance[u,v] for heavily used edges
        
        # Adaptive factor based on round number
        # Early rounds: smaller kick to allow exploration
        # Later rounds: larger kick to force diversification
        round_factor = 1.0 + 0.1 * _round_counter
        
        # Edge-specific factor based on usage
        # Edges used more often get a larger penalty
        usage_factor = 1.0 + 0.5 * (count / max_used)
        
        # Total kick multiplier
        kick = 0.3 * round_factor * usage_factor
        
        # Apply penalty to both directions (symmetric)
        modified_dist[u, v] += kick * edge_distance[u, v]
        modified_dist[v, u] += kick * edge_distance[v, u]
    
    # Ensure the matrix remains finite and symmetric
    modified_dist = np.maximum(modified_dist, 0)
    modified_dist = (modified_dist + modified_dist.T) / 2.0
    
    # Verify finiteness
    if not np.all(np.isfinite(modified_dist)):
        return edge_distance.copy()
    
    return modified_dist
# EVOLVE-BLOCK-END
