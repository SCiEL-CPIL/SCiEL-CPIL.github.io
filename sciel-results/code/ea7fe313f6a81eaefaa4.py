"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s hard limit (Task says 120s in contract, 150s in directive, use 120s to be safe)


def _greedy_size(adjacency, order):
    """Return the size of the independent set produced by a fixed greedy decoder."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v]:
            selected[v] = True
            count += 1
            # Mark neighbours as skipped (vectorized)
            selected[adjacency[v]] = True
    return count


def _degree(adjacency):
    return adjacency.sum(axis=1)


def _core_numbers(adjacency):
    """
    Compute the core number for each vertex.
    Uses the O(m+n) algorithm by Batagelj and Zaversnik.
    """
    n = adjacency.shape[0]
    deg = _degree(adjacency).astype(int)
    
    # Sort vertices by degree in descending order
    order = np.argsort(-deg)
    
    # Current degrees (will be modified)
    cur_deg = deg.copy()
    core = np.zeros(n, dtype=int)
    
    for v in order:
        if cur_deg[v] > core[v]:
            core[v] = cur_deg[v]
            # Update neighbors
            # Get neighbors of v
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                if cur_deg[u] > cur_deg[v]:
                    cur_deg[u] -= 1
                    
    return core


def _scores_hybrid(adjacency):
    """
    Combine Core Number, Degree, and Sum of Neighbor Degrees.
    We want to select Low Core, Low Degree, High SumNeighborDeg first.
    So Score should be high for these properties.
    Score = - (w1 * Core + w2 * Degree - w3 * SumNeighborDeg)
    """
    n = adjacency.shape[0]
    core = _core_numbers(adjacency)
    deg = _degree(adjacency).astype(float)
    sum_neighbor_deg = (adjacency @ deg)
    
    # Normalize features to [0, 1] to make weights comparable
    max_core = core.max() if core.max() > 0 else 1
    max_deg = deg.max() if deg.max() > 0 else 1
    max_snd = sum_neighbor_deg.max() if sum_neighbor_deg.max() > 0 else 1
    
    norm_core = core.astype(float) / max_core
    norm_deg = deg / max_deg
    norm_snd = sum_neighbor_deg / max_snd
    
    # Weights: Core is most important, then Degree, then SumNeighborDeg
    # High Core -> Bad -> Low Score
    # High Deg -> Bad -> Low Score
    # High SumNeighborDeg -> Good -> High Score
    w_core = 1.0
    w_deg = 0.5
    w_snd = 0.2
    
    score = -(w_core * norm_core + w_deg * norm_deg - w_snd * norm_snd)
    
    return score


def _local_search(adjacency, scores, max_iterations=1000):
    """
    Simple local search: try swapping adjacent elements in the priority order
    to see if it increases the independent set size.
    """
    n = scores.shape[0]
    order = np.argsort(-scores)
    current_size = _greedy_size(adjacency, order)
    best_size = current_size
    best_order = order.copy()
    
    # To speed up, we only consider swaps that might be beneficial.
    # A full 2-opt is O(n^2 * greedy_cost). Greedy cost is O(n+m).
    # n=200, m~2000. 40k * 2200 ops = 88M ops. Might be too slow in Python.
    # We will limit iterations.
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        for i in range(n - 1):
            # Check time
            if time.time() - _START_TIME > 110:
                return best_order, best_size
                
            # Swap i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _greedy_size(adjacency, new_order)
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
                # Continue searching from the new state
                break # Restart the loop with the new best_order
                
    return best_order, best_size


def priority(adjacency):
    """
    Priority function for Maximum Independent Set via greedy decoder.
    Returns one score per vertex; higher score = selected earlier.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    # Deadline guard
    def time_left():
        return 120.0 - 5.0 - (time.time() - _START_TIME)

    # 1. Generate strong static scores
    scores = _scores_hybrid(adjacency)
    
    # 2. Local Search to refine the order
    # Convert scores to order, then back to scores?
    # The interface requires returning scores.
    # We can return scores that correspond to the optimized order.
    # If we have an optimal order, we can assign scores based on rank in that order.
    
    order, size = _local_search(adjacency, scores, max_iterations=200)
    
    # Create new scores based on the optimized order
    # Higher rank (earlier in order) should have higher score
    new_scores = np.zeros(n)
    for i, v in enumerate(order):
        new_scores[v] = n - i
        
    return new_scores
# EVOLVE-BLOCK-END
