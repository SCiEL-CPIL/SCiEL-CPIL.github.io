"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_MARGIN = 10.0


def _cut_size(labels, adj):
    """Compute the cut size for a given labeling."""
    diff = (labels[None, :] != labels[:, None])
    return int(np.sum(adj * diff))


def _local_search(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain
    until no positive gain exists or deadline is reached.
    """
    n = len(adj)
    labels = labels.copy()
    row_sums = adj.sum(axis=1).astype(np.int32)
    
    # Compute current cut size
    cut = _cut_size(labels, adj)
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_v = -1
        
        for v in range(n):
            # Gain from flipping v: 2 * (neighbors on other side) - degree(v)
            # neighbors on other side = degree - neighbors on same side
            same = int(adj[v, labels == labels[v]].sum())
            deg = int(row_sums[v])
            gain = 2 * (deg - same) - deg
            if gain > best_gain:
                best_gain = gain
                best_v = v
        
        if best_v >= 0:
            labels[best_v] = 1 - labels[best_v]
            cut += best_gain
            improved = True
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return labels, cut


def partition(adjacency):
    """
    Find a good maximum cut partition using multiple random restarts with
    greedy local search, returning the best labeling found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    adj = np.asarray(adjacency, dtype=np.bool_)
    n = len(adj)
    
    # Ensure adjacency is symmetric (just in case)
    adj = (adj | adj.T)
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts we can afford; we'll loop until deadline
    np_rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Random initial labeling
        labels = np_rng.integers(0, 2, size=n)
        
        # Run local search with the deadline
        labels, cut = _local_search(labels, adj, deadline)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we used up the deadline inside local_search, break
        if time.time() >= deadline:
            break
    
    if best_labels is None:
        # Fallback: alternating labels
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
