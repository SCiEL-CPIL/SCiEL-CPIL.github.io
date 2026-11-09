import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder.
    
    Uses local greedy simulation: for each vertex v, simulate greedy selection
    where v is forced to be first (using degree-based order for the rest),
    and score v by the resulting independent set size.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    degree = A.sum(axis=1).astype(float)
    base_order = np.argsort(-degree)  # Higher degree first
    
    def greedy_decode_with_forced_first(forced_vertex, order):
        """
        Simulate greedy decoder where forced_vertex is selected first.
        Returns the size of the resulting independent set.
        """
        selected = np.zeros(n, dtype=bool)
        selected[forced_vertex] = True
        count = 1
        
        # Block neighbors of forced_vertex
        neighbors = A[forced_vertex]
        for u in np.where(neighbors)[0]:
            selected[u] = True  # Mark as blocked (won't be selected)
        
        for v in order:
            if v == forced_vertex:
                continue
            if selected[v]:
                continue
            # Check if v has any selected neighbor
            if np.dot(selected, A[v]):
                continue
            selected[v] = True
            count += 1
            # Block neighbors
            for u in np.where(A[v])[0]:
                if not selected[u]:
                    selected[u] = True
        
        return count
    
    scores = np.zeros(n)
    
    # Evaluate each vertex as forced-first
    for v in range(n):
        if time.time() > deadline - 5.0:
            break
        scores[v] = greedy_decode_with_forced_first(v, base_order)
    
    # If we ran out of time, fill remaining with degree-based score
    if time.time() > deadline - 5.0:
        remaining = np.where(scores == 0)[0]
        if len(remaining) > 0:
            scores[remaining] = -degree[remaining]
    
    # Higher score = better (more vertices in IS when this vertex is first)
    return scores

# EVOLVE-BLOCK-END
