"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Returns a priority score for each vertex.
    Higher score means higher priority (selected first in greedy decoder).
    
    Strategy:
    Prioritize vertices that have high-degree neighbors (hubs) but low degree themselves.
    This allows the greedy algorithm to "block" high-degree hubs by selecting their 
    low-degree neighbors, which is more efficient than selecting the hubs directly 
    (which would block many other vertices).
    
    Score = Average Neighbor Degree - Own Degree
    """
    start_time = time.time()
    
    # Convert to float for calculations if not already
    A = np.asarray(adjacency, dtype=float)
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate sum of degrees of neighbors for each vertex
    # S[i] = sum(degrees[j] for j in neighbors of i)
    neighbor_degree_sums = A @ degrees
    
    # Avoid division by zero for isolated vertices
    # Isolated vertices have degree 0. Their score should be high (good to pick) 
    # because they don't block anyone.
    # Formula: (Sum / Degree) - Degree
    # If Degree is 0, we treat Average Neighbor Degree as 0. Score = 0 - 0 = 0?
    # Actually, an isolated vertex is the best candidate to pick. It costs 1 node, blocks 0 others.
    # A vertex with degree 1 connected to a hub (deg 100) has score 100 - 1 = 99.
    # An isolated vertex has score 0.
    # Should isolated vertices have higher priority than satellites of hubs?
    # Picking an isolated vertex: +1 to set.
    # Picking a satellite of a hub: +1 to set, removes hub.
    # Both give +1 to set size immediately. But removing the hub prevents the hub from being picked later (which would be bad) or prevents the hub from blocking others?
    # If the hub is not picked, it still blocks its neighbors.
    # Wait, if I pick the satellite, the hub is removed from the graph. The hub's other neighbors are NOT removed.
    # If I pick the isolated vertex, the isolated vertex is removed.
    # In both cases, 1 vertex added to IS.
    # However, picking the satellite "saves" the hub's other neighbors from being blocked by the hub? No, the hub is gone, so it can't block anyone.
    # So picking the satellite is strictly better or equal to picking the isolated vertex, because it also neutralizes the hub.
    # So satellites of hubs should have higher priority than isolated vertices.
    # My formula: Satellite (99) > Isolated (0). This holds.
    
    # Calculate Average Neighbor Degree
    # Use np.divide with where clause to handle zeros
    avg_neighbor_degrees = np.zeros_like(degrees)
    mask = degrees > 0
    avg_neighbor_degrees[mask] = neighbor_degree_sums[mask] / degrees[mask]
    
    # Score: Higher is better.
    # We want high avg_neighbor_degree and low own_degree.
    scores = avg_neighbor_degrees - degrees
    
    # Safety check for time (though this is very fast for N=200)
    elapsed = time.time() - start_time
    if elapsed > 110: # Safety margin
        # Fallback to simple degree if somehow slow (unlikely)
        return -degrees.astype(float)
        
    return scores.astype(float)
# EVOLVE-BLOCK-END
