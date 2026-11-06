"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    The greedy decoder picks vertices in descending score order, skipping neighbors
    of already-picked vertices. We want scores that lead to large independent sets.
    
    Strategy: Use a combination of structural heuristics with a fast vectorized simulation
    to select the best scoring strategy from a set of candidates.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Convert to boolean for efficient operations
    adj_bool = adj.astype(bool)
    
    # Compute degree
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Compute for each vertex: the sum of degrees of its neighbors
    neighbor_degree_sum = (adj_bool @ degree).astype(float)
    
    # Compute for each vertex: the number of low-degree neighbors (degree <= median)
    median_degree = np.median(degree)
    low_degree_neighbors = (adj_bool @ (degree <= median_degree).astype(float)).astype(float)
    
    # Normalize the components to comparable scales
    max_degree = degree.max() if degree.max() > 0 else 1.0
    max_neighbor_degree_sum = neighbor_degree_sum.max() if neighbor_degree_sum.max() > 0 else 1.0
    max_low_degree_neighbors = low_degree_neighbors.max() if low_degree_neighbors.max() > 0 else 1.0
    
    # Strategy: We want to pick vertices that:
    # 1. Have low degree (fewer conflicts)
    # 2. Are connected to high-degree vertices (their high-degree neighbors will be skipped)
    # 3. Have few low-degree neighbors (avoid picking vertices whose low-degree neighbors would also be picked)
    
    # Weighted combination
    score = (-degree / max_degree) + (neighbor_degree_sum / max_neighbor_degree_sum) - (low_degree_neighbors / max_low_degree_neighbors)
    
    # Check if we're still within time budget
    if time.time() > deadline:
        return score
    
    def simulate_greedy_fast(scores):
        """Simulate the greedy decoder using vectorized operations and return the size of the independent set."""
        selected = 0
        available = np.ones(n, dtype=bool)
        
        # Get vertices in order of descending score
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            # Select this vertex
            selected += 1
            # Remove it and its neighbors from available
            available[v] = False
            neighbors = np.where(adj_bool[v])[0]
            if len(neighbors) > 0:
                available[neighbors] = False
            
            # Early exit optimization: if no more available vertices
            if not available.any():
                break
        
        return selected
    
    # Generate a few alternative scoring strategies
    strategies = []
    
    # Strategy 1: Current composite score
    strategies.append(score)
    
    # Strategy 2: Pure degree-based (original seed)
    strategies.append(-degree)
    
    # Strategy 3: Degree-based with neighbor degree adjustment
    strategies.append(-degree + neighbor_degree_sum / (max_neighbor_degree_sum + 1e-10))
    
    # Strategy 4: Inverse degree (higher for low degree)
    strategies.append(1.0 / (degree + 1.0))
    
    # Strategy 5: Degree-based with low-degree neighbor penalty
    strategies.append(-degree - low_degree_neighbors / (max_low_degree_neighbors + 1e-10))
    
    # Strategy 6: Pure neighbor degree sum (prefer vertices next to hubs)
    strategies.append(neighbor_degree_sum)
    
    # Strategy 7: Combined: low degree + high neighbor degree
    strategies.append(-degree / max_degree + neighbor_degree_sum / max_neighbor_degree_sum)
    
    # Strategy 8: Low degree with tie-break by neighbor degree
    strategies.append(-degree - 0.1 * neighbor_degree_sum / max_neighbor_degree_sum)
    
    # Evaluate each strategy and pick the best
    best_score = None
    best_size = -1
    
    for strat in strategies:
        if time.time() > deadline:
            break
        size = simulate_greedy_fast(strat)
        if size > best_size:
            best_size = size
            best_score = strat
    
    if best_score is not None:
        return best_score
    else:
        return score
# EVOLVE-BLOCK-END
