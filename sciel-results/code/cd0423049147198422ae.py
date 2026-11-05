"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Use a deadline guard: aim to finish well under 120s.
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute adjacency as a boolean numpy array for fast operations
    A = np.asarray(adjacency, dtype=bool)
    
    # Helper to compute cut size efficiently
    def cut_size(labels):
        # Cut edges are those where labels differ
        # Using matrix multiplication: cut = sum(A * (labels[:,None] != labels[None,:]))
        # But that's O(n^2) memory. For n=100 it's fine.
        # Alternative: for each vertex, count neighbors with different label.
        # cut = sum over v of (A[v] & (labels != labels[v])).sum()
        # Vectorized: 
        diff = labels[:, None] != labels[None, :]
        return (A & diff).sum() // 2  # Each edge counted twice
    
    # Strategy: multiple random restarts with local search (flip if it improves cut)
    best_labels = None
    best_cut = -1
    
    # Number of restarts we can afford
    # Each local search pass is O(n^2) ~ 10^4 ops, very fast.
    # We can do many restarts.
    
    # Start with a good heuristic: random initialization, then local search
    # Also try some structured initializations
    
    num_restarts = 50  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random
            labels = np.random.randint(0, 2, n)
        elif restart == 2:
            # Random with bias
            labels = np.random.randint(0, 2, n)
        else:
            # Random
            labels = np.random.randint(0, 2, n)
        
        # Local search: flip vertices that improve the cut
        # For each vertex v, if flipping it increases the cut, do it.
        # Gain from flipping v:
        #   If v is in group 0: currently contributes (edges from v to group 1) to cut.
        #   After flip: contributes (edges from v to group 0) to cut.
        #   Net gain = (edges to group 0) - (edges to group 1)
        #   = 2*(edges to group 0) - degree(v)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to break ties
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                # Compute gain from flipping v
                # edges_to_same = A[v, labels == labels[v]].sum()
                # edges_to_diff = A[v, labels != labels[v]].sum()
                # gain = edges_to_same - edges_to_diff
                # If gain > 0, flipping improves the cut
                same_mask = (labels == labels[v])
                diff_mask = (labels != labels[v])
                edges_to_same = A[v, same_mask].sum()
                edges_to_diff = A[v, diff_mask].sum()
                gain = edges_to_same - edges_to_diff
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        # Evaluate
        cur_cut = cut_size(labels)
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
