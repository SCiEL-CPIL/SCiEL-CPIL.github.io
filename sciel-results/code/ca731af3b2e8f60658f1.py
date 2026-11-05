"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for greedy Maximum Independent Set selection.
    
    Strategy: Dynamic exclusion simulation captures interference patterns.
    Vertices in low-density regions become eligible earlier and get higher priority.
    We simulate the greedy process with a time-aware scoring that accounts for
    the expected order in which vertices will be excluded by their neighbors.
    
    For each vertex, we estimate how early it would be picked in a degree-based
    greedy process, and how many of its neighbors would also be picked early,
    causing it to be excluded. Vertices that are "safe" (few neighbors that
    would be picked before them) get higher scores.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=float)
    
    # Convert to boolean array if needed
    adj = np.asarray(adjacency, dtype=bool)
    
    # Degree of each vertex
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Compute a "local density" measure: for each vertex, how crowded is its neighborhood?
    # Second-order degree: sum of degrees of neighbors
    second_deg = (adj @ degrees).astype(np.float64)
    
    # Normalize: local density = second_deg / max(1, degrees)
    # This gives the average degree of neighbors
    avg_neighbor_deg = np.where(degrees > 0, second_deg / np.maximum(degrees, 1), 0.0)
    
    # A vertex with low degree AND low average neighbor degree is in a "sparse" region
    # and should be prioritized (will be safe from exclusion)
    
    # Simulate a simplified dynamic exclusion to estimate when each vertex would be excluded
    # Key insight: if we pick vertices in order of (degree, then random tiebreak),
    # a vertex v gets excluded if any of its neighbors is picked before v.
    # The probability that a neighbor u is picked before v depends on their relative scores.
    
    # Use an iterative approach: estimate the "effective priority" considering exclusion
    # Start with degree-based priority
    scores = -degrees.copy()
    
    # Iterate: for each vertex, adjust its score based on how many neighbors
    # have higher priority (i.e., would be picked before it and exclude it)
    for iteration in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Sort vertices by current scores (descending)
        order = np.argsort(-scores)
        
        # For each vertex, count how many neighbors appear earlier in the order
        # (i.e., would be picked before it and cause exclusion)
        # We do this efficiently using the adjacency matrix
        excluded_count = np.zeros(n, dtype=np.float64)
        
        # Process in order: when we "pick" a vertex, all its unpicked neighbors get excluded
        picked = np.zeros(n, dtype=bool)
        for v in order:
            if time.time() > deadline - 5.0:
                break
            if picked[v]:
                continue
            # v is picked; its neighbors will be excluded
            neighbors = np.where(adj[v])[0]
            for u in neighbors:
                if not picked[u]:
                    excluded_count[u] += 1.0
            picked[v] = True
        
        # Update scores: vertices with fewer exclusions get higher scores
        # Combine with original degree-based score
        # Use a weighted combination
        alpha = 0.7
        beta = 0.3
        scores = alpha * (-degrees) + beta * (-excluded_count)
        
        # Add a small perturbation based on second-order structure
        # Vertices whose neighbors are also low-degree are safer
        scores -= 0.1 * avg_neighbor_deg
    
    # Final adjustment: ensure we return a valid score vector
    # Add tiny random perturbation to break ties deterministically
    rng = np.random.default_rng(42)
    scores += rng.uniform(0, 1e-10, size=n)
    
    return scores
# EVOLVE-BLOCK-END
