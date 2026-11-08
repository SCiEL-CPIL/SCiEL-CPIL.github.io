# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # adjacency is a boolean matrix, labels is 0/1 array
    # An edge (i,j) crosses if adjacency[i,j] and labels[i] != labels[j]
    # Sum over all i,j where adjacency is True and labels differ, divide by 2
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    # Current contribution of v to cut: number of neighbors with opposite label
    # If we flip v, its contribution becomes: number of neighbors with same label (old)
    # Gain = new_contribution - old_contribution
    # old_contribution = sum of adjacency[v][labels != labels[v]]
    # new_contribution = sum of adjacency[v][labels == labels[v]]
    # Gain = new - old = (sum_adj_same) - (sum_adj_opp)
    # But note: flipping v also changes the contribution of all neighbors of v.
    # Actually, the standard way: the gain in cut size from flipping v is:
    # For each neighbor u of v:
    #   If u and v currently differ, the edge was cut. After flip, they are same, so edge no longer cut. Loss 1.
    #   If u and v currently same, the edge was not cut. After flip, they differ, so edge now cut. Gain 1.
    # So gain = (# neighbors with same label) - (# neighbors with opposite label)
    # Wait, let me re-derive.
    # Let S = set of neighbors of v.
    # Current cut edges incident to v: edges (v,u) where u in S and labels[u] != labels[v]
    # After flipping v: cut edges incident to v: edges (v,u) where u in S and labels[u] == labels[v] (old)
    # So change = (new count) - (old count) = (# u in S with labels[u] == labels[v]) - (# u in S with labels[u] != labels[v])
    # This is the gain.
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _local_search(adjacency, labels, deadline):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for v in range(n):
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


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
    deadline = start_time + 110  # Safety margin: 110s out of 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different random initializations
    # Also try the alternating initialization
    num_restarts = 10
    base_seed = 42
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Alternating start
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random start with a different seed
            labels = _random_labels(n, base_seed + restart)
        else:
            labels = _random_labels(n, base_seed + restart)
        
        # Run local search
        labels = _local_search(adjacency, labels, deadline)
        
        # Evaluate
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
