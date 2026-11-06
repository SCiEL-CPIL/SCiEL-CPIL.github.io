# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Vectorized Best-Improvement Local Search.
    Combines fast vectorized gain computation with multi-start restarts.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    E = 0.5 * adj.sum()  # total number of edges
    degrees = adj.sum(axis=1)

    # Deadline guard: 145 seconds (5s safety margin from 150s limit)
    deadline = time.time() + 145.0

    def get_cut_size(labels):
        """Compute the number of edges crossing the partition."""
        Z = 2.0 * labels - 1.0
        temp = adj @ Z
        zadjz = Z @ temp
        cut = 0.25 * (2.0 * E - zadjz)
        return cut

    def local_search(labels, max_iterations=200):
        """
        Vectorized best-improvement local search.
        For each vertex, compute the gain of flipping it, and flip the vertex
        with the maximum positive gain. This avoids the O(n^2) per-vertex scan
        of standard greedy and converges faster.
        """
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            
            # Compute gains for all vertices at once using vectorized operations
            # gain[v] = (edges to opposite side) - (edges to same side)
            # If labels[v] == 0: same = neighbors with label 0, diff = neighbors with label 1
            # If labels[v] == 1: same = neighbors with label 1, diff = neighbors with label 0
            
            # diff_count[v] = sum of adj[v, u] * labels[u] for u where labels[u] != labels[v]
            # Actually, let's use a cleaner formulation:
            # Let Z = 2*labels - 1, so Z[v] is +1 or -1
            # gain[v] = sum_u adj[v,u] * (Z[u] * Z[v]) ... no, that's not right.
            
            # Let's think again:
            # If we flip v, the change in cut is:
            # For each neighbor u:
            #   If labels[u] == labels[v], the edge (v,u) was NOT crossing, now it IS -> +1
            #   If labels[u] != labels[v], the edge (v,u) WAS crossing, now it IS NOT -> -1
            # So gain[v] = (# neighbors with same label) - (# neighbors with different label)
            
            # Using vectorized ops:
            # same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
            # diff_count[v] = sum_u adj[v,u] * (labels[u] != labels[v])
            
            # We can compute:
            # same_count = adj @ (labels == labels[:, None]).astype(float)
            # But that's O(n^2) memory. Instead:
            
            # For labels[v] == 0: same_count[v] = sum_u adj[v,u] * (1 - labels[u])
            # For labels[v] == 1: same_count[v] = sum_u adj[v,u] * labels[u]
            
            # Let's use: same_count = adj @ (1 - labels) * (labels == 0) + adj @ labels * (labels == 1)
            # This is still not fully vectorized cleanly. Let's use a different approach.
            
            # Z = 2*labels - 1
            # gain[v] = sum_u adj[v,u] * Z[u] * Z[v] ... no.
            
            # Actually: gain[v] = 2 * (sum_u adj[v,u] * labels[u]) * Z[v] - degrees[v]
            # Let's verify:
            # If Z[v] = 1 (labels[v]=1): gain = 2 * (adj @ labels)[v] - degrees[v]
            #   adj @ labels)[v] = # neighbors with label 1
            #   gain = 2 * (# same) - degrees = # same - # diff. Correct.
            # If Z[v] = -1 (labels[v]=0): gain = 2 * (adj @ labels)[v] * (-1) - degrees[v]
            #   = -2 * (# neighbors with label 1) - degrees[v]
            #   = -2 * (# diff) - (# same + # diff)
            #   = -# same - 3 * # diff. That's wrong.
            
            # Let me re-derive:
            # Let s = # neighbors with same label, d = # neighbors with different label
            # gain = s - d
            # s + d = degrees[v]
            # So gain = 2s - degrees[v] or gain = degrees[v] - 2d
            
            # For labels[v] = 0: s = # neighbors with label 0 = sum_u adj[v,u] * (1-labels[u])
            #   = degrees[v] - (adj @ labels)[v]
            #   gain = 2*(degrees[v] - (adj@labels)[v]) - degrees[v] = degrees[v] - 2*(adj@labels)[v]
            
            # For labels[v] = 1: s = # neighbors with label 1 = (adj @ labels)[v]
            #   gain = 2*(adj@labels)[v] - degrees[v]
            
            # So: gain[v] = Z[v] * (2*(adj@labels)[v] - degrees[v])
            # Let's verify:
            # Z[v]=1: gain = 2*(adj@labels)[v] - degrees[v]. Correct.
            # Z[v]=-1: gain = -(2*(adj@labels)[v] - degrees[v]) = degrees[v] - 2*(adj@labels)[v]. Correct.
            
            Z = 2.0 * labels - 1.0
            adj_labels = adj @ labels  # (adj @ labels)[v] = sum_u adj[v,u] * labels[u]
            gains = Z * (2.0 * adj_labels - degrees)
            
            # Find the vertex with maximum positive gain
            max_gain = gains.max()
            if max_gain <= 0:
                break
            
            # Flip the vertex with maximum gain
            v = int(np.argmax(gains))
            labels[v] = 1 - labels[v]
            current_cut += max_gain
        
        return labels, current_cut

    # Spectral initialization using the Fiedler vector
    def spectral_init():
        """Create initial partition using the Fiedler vector."""
        D = sp.diags(degrees, format='csr')
        A = sp.csr_matrix(adj)
        L = D - A
        
        try:
            evals, evecs = spla.eigsh(L, k=2, sigma=0, which='LM', tol=1e-8)
            idx = np.argsort(evals)
            fiedler = evecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(n)
        
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        return labels

    # Multi-start local search
    best_labels = None
    best_cut = -1

    # Start with spectral initialization
    labels = spectral_init()
    labels, cut = local_search(labels, max_iterations=300)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        labels = rng.integers(0, 2, size=n).astype(int)
        labels, cut = local_search(labels, max_iterations=200)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        labels, cut = local_search(best_labels, max_iterations=500)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
