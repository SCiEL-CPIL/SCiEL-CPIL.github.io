# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _walksat(adjacency, labels, deadline, max_iters=20000):
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    
    # Precompute total degrees
    total_degrees = adjacency.sum(axis=1)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        best_gain = np.max(gains)
        
        if best_gain > 0:
            # Flip the vertex with the best gain
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            cut += best_gain
        else:
            # Local optimum. Pick a vertex involved in an uncut edge.
            uncut_mask = adjacency & (~diff)
            uncut_edges = np.argwhere(uncut_mask)
            
            if len(uncut_edges) == 0:
                break
                
            # Pick a random uncut edge
            edge_idx = np.random.randint(0, len(uncut_edges))
            u, v = uncut_edges[edge_idx]
            
            # Flip one of them randomly
            if np.random.rand() < 0.5:
                flip_v = u
            else:
                flip_v = v
                
            # Update cut
            cut += gains[flip_v]
            labels[flip_v] = 1 - labels[flip_v]
            
    return labels, cut


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different random initializations
    num_restarts = 20  # Increase restarts since WalkSAT is fast per step
    base_seed = 42
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Alternating start
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random start
            labels = _random_labels(n, base_seed + restart)
        else:
            labels = _random_labels(n, base_seed + restart)
        
        # Run WalkSAT
        labels, cut = _walksat(adjacency, labels, deadline)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
