import numpy as np
import time

def _greedy_decode(adj, order):
    """Simulate greedy decoding with a given vertex order. Returns set size."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            blocked[adj[v]] = True
    return count

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Use local search to optimize vertex ordering directly. Start with a degree-based
    heuristic, then iteratively swap adjacent vertices in the ordering if it improves
    the greedy decoder's output.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(float)
    
    # Initial ordering: sort by degree (lowest first - LDF heuristic)
    order = np.argsort(deg, kind='stable')
    current_size = _greedy_decode(adj, order)
    
    # Local search: try swaps of adjacent elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 10.0:
                break
            # Try swapping order[i] and order[i+1]
            v1, v2 = order[i], order[i+1]
            order[i], order[i+1] = v2, v1
            
            new_size = _greedy_decode(adj, order)
            if new_size > current_size:
                current_size = new_size
                improved = True
            else:
                # Revert
                order[i], order[i+1] = v1, v2
    
    # Convert ordering to scores: vertices earlier in order get higher scores
    # Use rank-based scoring: vertex at position i gets score (n - i)
    scores = np.zeros(n)
    for rank, v in enumerate(order):
        scores[v] = n - rank
    
    return scores
