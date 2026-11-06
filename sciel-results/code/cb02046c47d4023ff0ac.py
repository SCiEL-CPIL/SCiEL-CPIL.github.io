"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge crosses if endpoints have different labels
    # cut = sum over all edges (i,j) of (labels[i] != labels[j])
    # For a boolean adjacency matrix, this is:
    # sum of adj[i,j] where labels[i] != labels[j]
    # Equivalent: 0.5 * sum over all i,j of adj[i,j] * (labels[i] != labels[j])
    # But since adj is symmetric, we can compute:
    # For each vertex v, count edges to vertices with different label
    # cut = 0.5 * sum_v (deg[v] - same_label_neighbors[v])
    # Simpler: use matrix multiplication
    # labels as +1/-1
    s = 2 * labels - 1  # +1/-1
    # cut = 0.5 * sum_{i,j} adj[i,j] * (s[i] != s[j])
    # = 0.5 * (sum_{i,j} adj[i,j] - sum_{i,j} adj[i,j]*s[i]*s[j])
    # = 0.5 * (2*E - s^T A s) where E = total edges
    # But easier: just compute directly
    diff = (labels[:, None] != labels[None, :])
    return int(np.sum(adj * diff))


def _local_search(adj, labels, max_passes=50):
    """Vectorized single-flip local search. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        # For each vertex, compute gain if we flip
        # If vertex v is in group 0, flipping to group 1 changes cut by:
        #   (neighbors in group 1) - (neighbors in group 0)
        #   = 2*(neighbors in group 1) - deg[v]
        # If vertex v is in group 1, flipping to group 0 changes cut by:
        #   (neighbors in group 0) - (neighbors in group 1)
        #   = 2*(neighbors in group 0) - deg[v]
        
        # Vectorized: for each v, count neighbors with same label
        same_label = (labels[:, None] == labels[None, :])
        same_count = np.sum(adj * same_label, axis=1)
        deg = np.sum(adj, axis=1)
        
        # Gain for flipping vertex v:
        # If v is in group 0: gain = (deg[v] - same_count[v]) - same_count[v] = deg[v] - 2*same_count[v]
        # If v is in group 1: gain = same_count[v] - (deg[v] - same_count[v]) = 2*same_count[v] - deg[v]
        # Unified: gain[v] = (2*same_count[v] - deg[v]) if labels[v]==1 else (deg[v] - 2*same_count[v])
        # = (1 - 2*labels[v]) * (deg[v] - 2*same_count[v]) ... let me re-derive
        
        # Let's define: for vertex v, let s_v = number of neighbors in same group
        # Let d_v = degree of v
        # If v is in group 0:
        #   Currently contributes s_v to "non-cut" (edges within group 0)
        #   If flipped to group 1: contributes (d_v - s_v) to "non-cut" from group 1 side
        #   Change in cut = (d_v - s_v) - s_v = d_v - 2*s_v ... wait, that's not right
        #   
        # Let me think more carefully:
        # Cut size = number of edges with endpoints in different groups
        # For vertex v in group 0:
        #   Edges from v that are in cut = number of neighbors in group 1 = d_v - s_v
        # If we flip v to group 1:
        #   Edges from v that are in cut = number of neighbors in group 0 = s_v
        # Change in cut (from v's perspective) = s_v - (d_v - s_v) = 2*s_v - d_v
        # But we also need to account for the fact that edges from OTHER vertices to v change
        # Actually no - the cut size is symmetric. If we flip v, the change in total cut is:
        #   For each neighbor u of v:
        #     If u is in same group as v: edge (u,v) was NOT in cut, now IS in cut → +1
        #     If u is in different group: edge (u,v) WAS in cut, now NOT in cut → -1
        #   So change = s_v - (d_v - s_v) = 2*s_v - d_v
        #
        # So gain[v] = 2*same_count[v] - deg[v] if v is in group 0
        #            = -(2*same_count[v] - deg[v]) if v is in group 1 ... 
        #
        # Wait, let me re-derive for v in group 1:
        #   Edges from v in cut = neighbors in group 0 = d_v - s_v
        #   If flipped to group 0: edges in cut = neighbors in group 1 = s_v
        #   Change = s_v - (d_v - s_v) = 2*s_v - d_v
        #
        # So in both cases, gain = 2*same_count[v] - deg[v]
        # We flip if gain > 0
        
        gain = 2 * same_count - deg
        # Flip all vertices with positive gain
        flip_mask = gain > 0
        if np.any(flip_mask):
            labels[flip_mask] = 1 - labels[flip_mask]
            improved = True
        
        if not improved:
            break
    
    return labels, _cut_size(adj, labels)


def _spectral_init(adj, rng):
    """Generate initial labels using Fiedler vector (second eigenvector of Laplacian)."""
    n = len(adj)
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector with the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Partition: vertices with positive Fiedler value → group 1, negative → group 0
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to random
        labels = rng.integers(0, 2, size=n).astype(np.int8)
    
    return labels


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    rng = np.random.default_rng(42)
    
    # First, try spectral initialization
    try:
        init_labels = _spectral_init(adj, rng)
        labels, cut = _local_search(adj, init_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    except Exception:
        pass
    
    # Multiple restarts: spectral init + random perturbation
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial labels
        if r == 0:
            # Already did spectral init above
            continue
        
        # Mix of spectral and random
        if r % 3 == 0:
            # Pure random
            init_labels = rng.integers(0, 2, size=n).astype(np.int8)
        elif r % 3 == 1:
            # Spectral with slight perturbation
            try:
                init_labels = _spectral_init(adj, rng)
                # Perturb a few vertices
                n_perturb = max(1, n // 10)
                perturb_idx = rng.choice(n, size=n_perturb, replace=False)
                init_labels[perturb_idx] = 1 - init_labels[perturb_idx]
            except Exception:
                init_labels = rng.integers(0, 2, size=n).astype(np.int8)
        else:
            # Random
            init_labels = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Run local search
        labels, cut = _local_search(adj, init_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final polish: run local search a few more times from best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            labels, cut = _local_search(adj, best_labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
