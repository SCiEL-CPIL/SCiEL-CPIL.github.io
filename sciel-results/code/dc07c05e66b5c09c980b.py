<<<<<<< SEARCH
import numpy as np
import time


def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # An edge (i, j) crosses if labels[i] != labels[j].
    # Since adjacency is symmetric and unweighted, we count each crossing edge once.
    # sum over i < j where labels[i] != labels[j] and adjacency[i,j] is True.
    # Vectorized:
    diff = labels[:, None] != labels[None, :]
    # Only count upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    crossings = diff & triu_mask
    return int((crossings & adjacency).sum())


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses a time limit to stop before exceeding the budget.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adjacency)
    
    # Precompute row sums for efficiency? Actually, we can compute gain for flipping vertex v.
    # Gain = (edges to opposite set) - (edges to same set) = 2 * edges_to_opposite - total_degree
    # Wait, if we flip v:
    # Old: edges to opposite are cut, edges to same are not cut.
    # New: edges to opposite are NOT cut, edges to same ARE cut.
    # Change in cut = -edges_to_opposite + edges_to_same = edges_to_same - edges_to_opposite.
    # So we flip if edges_to_same > edges_to_opposite.
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            same_count = int(adjacency[v][labels == labels[v]].sum())
            opposite_count = int(adjacency[v][labels != labels[v]].sum())
            if same_count > opposite_count:
                labels[v] = 1 - labels[v]
                current_cut += same_count - opposite_count
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut
=======
import numpy as np
import time


def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    diff = labels[:, None] != labels[None, :]
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    crossings = diff & triu_mask
    return int((crossings & adjacency).sum())


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search with efficient gain computation.
    Uses incremental updates to avoid recomputing sums from scratch.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adjacency)
    
    # Precompute adjacency as numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # gain[v] = edges_to_same - edges_to_opposite
    # If gain[v] > 0, flipping v improves the cut
    # edges_to_same = adj[v] @ labels
    # edges_to_opposite = degrees[v] - edges_to_same
    # gain = edges_to_same - (degrees[v] - edges_to_same) = 2*edges_to_same - degrees[v]
    
    # Compute initial gains
    edges_to_same = adj @ labels
    gains = 2 * edges_to_same - degrees
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            if gains[v] > 0:
                # Flip vertex v
                old_label = labels[v]
                new_label = 1 - old_label
                labels[v] = new_label
                current_cut += gains[v]
                improved = True
                
                # Update gains for all neighbors of v
                # When v flips, for each neighbor u:
                #   if u has same label as new v (was opposite), edges_to_same[u] decreases by 1
                #   if u has same label as old v (was same), edges_to_same[u] increases by 1
                # Simpler: for each neighbor u, edges_to_same[u] changes by (new_label - old_label)
                # which is ±1
                delta = new_label - old_label  # +1 or -1
                neighbors = np.where(adj[v] != 0)[0]
                # For each neighbor u, edges_to_same[u] changes by delta * adj[v,u]
                # Since adj[v,u] is 1 for neighbors, edges_to_same[u] += delta
                # So gains[u] += 2 * delta
                gains[neighbors] += 2 * delta
        
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut
>>>>>>> REPLACE

