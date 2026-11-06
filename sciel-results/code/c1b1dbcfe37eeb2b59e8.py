# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using vectorized local search with multiple restarts.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    deg = adj.sum(axis=1)
    
    best_labels = None
    best_cut = -1
    
    def calc_cut(labels):
        # labels is 0/1
        # Convert to +/-1
        x = 2 * labels - 1
        val = x @ (adj @ x)
        total_edges = np.sum(adj) // 2
        return total_edges - 0.5 * val

    def local_search(labels):
        """
        Vectorized local search: flip all nodes that improve cut in current state.
        Keeps track of the best state encountered.
        """
        labels = labels.copy()
        current_cut = calc_cut(labels)
        best_local_labels = labels.copy()
        best_local_cut = current_cut
        
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Compute n1[v] = number of neighbors in set 1
            n1 = adj @ labels
            # D = n0 - n1 = deg - 2*n1
            D = deg - 2 * n1
            
            # Determine which nodes to flip
            # If labels[v] == 0, flip if D > 0 (gain is D)
            # If labels[v] == 1, flip if D < 0 (gain is -D)
            flip_mask = ((labels == 0) & (D > 0)) | ((labels == 1) & (D < 0))
            
            if not np.any(flip_mask):
                break
            
            # Apply flips
            labels[flip_mask] = 1 - labels[flip_mask]
            
            # Evaluate new cut
            current_cut = calc_cut(labels)
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
            
            # If we didn't improve the best, and we are stuck in a loop (same cut or worse), 
            # we might want to break if we see the same state, but for now 
            # the "flip all positive" heuristic is robust enough for N=100.
            # If current_cut <= best_local_cut for several iterations, we could break, 
            # but checking every iter is fine.
            
        return best_local_labels

    # Multiple random restarts
    rng = np.random.default_rng()
    
    # Start with alternating pattern
    labels = np.arange(n) % 2
    labels = local_search(labels)
    c = calc_cut(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Try random restarts
    while time.time() - start_time < DEADLINE:
        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        labels = local_search(labels)
        c = calc_cut(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
