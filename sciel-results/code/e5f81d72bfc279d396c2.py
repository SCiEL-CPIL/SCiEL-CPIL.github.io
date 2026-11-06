# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    # Crossing edges: label[v] != label[u]. Using adjacency matrix:
    # cut = sum over edges (1 - (2*label[v]-1)*(2*label[u]-1)) / 2
    # Simpler: cut = 0.5 * (sum of all edges - sum of same-side edges)
    # Using matrix form: let s = 2*labels - 1 (so ±1). cut = 0.5*(E_total - s^T A s)/2 ... 
    # Actually: cut = 0.25 * (E_total + s^T (D - A) s) ... let me just do it directly.
    s = 2 * labels.astype(np.float64) - 1  # ±1
    # Number of edges with same label: s^T A s counts 2 for each same-side edge (since s_v*s_u=1)
    # and 0 for each crossing edge (since s_v*s_u=-1). Wait: s^T A s = sum_v sum_u s_v A_vu s_u
    # = sum over edges (u,v) of (s_u + s_v)*... no, A is symmetric 0/1.
    # s^T A s = sum_{v,u} s_v A_{vu} s_u = 2 * sum_{u<v, A_uv=1} s_u s_v
    # For same-side edge: s_u s_v = 1. For crossing: s_u s_v = -1.
    # So s^T A s = 2*(#same - #cross)
    # Total edges E = #same + #cross
    # => #cross = (E - s^T A s / 2) / 2 = E/2 - s^T A s / 4
    E_total = 0.5 * adj.sum()
    s_As = s @ (adj @ s)
    cut = E_total - 0.25 * s_As
    return int(round(cut))


def _greedy_refine(adj, labels, max_passes=50):
    """
    Flip-based greedy local search. Each pass, for each vertex, compute the
    net change in cut size if we flip it, and flip if beneficial.
    Uses incremental updates for efficiency.
    """
    n = len(labels)
    labels = labels.copy()
    # s[v] = 2*labels[v] - 1, in {-1, +1}
    s = 2 * labels.astype(np.float64) - 1
    
    # gain[v] = change in cut size if we flip vertex v
    # Flipping v changes s[v] -> -s[v].
    # The contribution of v to s^T A s involves: 2 * s[v] * sum_u A[v,u] s[u]
    # Let's derive: cut = E/2 - (1/4) s^T A s
    # d(cut)/d(flip v): new s^T A s = old - 2 * s[v] * (A s)[v] * 2 ... 
    # Actually, flipping v: s[v] -> -s[v]. The term s^T A s changes by:
    # (A s)[v] is sum_u A[v,u] s[u]. The contribution involving s[v] is:
    # s[v] * (A s)[v] + s[v] * (A^T s)[v] = 2 s[v] (A s)[v] (since A symmetric)
    # Wait, s^T A s = sum_v s[v] (A s)[v]. Flipping v changes s[v] to -s[v],
    # so the change is: -s[v](As)[v] - s[v](As)[v] = -2 s[v] (As)[v]
    # New s^T A s = old - 2 s[v] (As)[v]
    # Change in cut = -(1/4) * change in (s^T A s) = -(1/4) * (-2 s[v] (As)[v]) = (1/2) s[v] (As)[v]
    # So gain[v] = 0.5 * s[v] * (A s)[v]
    
    # Compute (A s) once
    As = adj @ s
    
    # gain[v] = 0.5 * s[v] * As[v]
    gain = 0.5 * s * As
    
    # Greedy passes
    for _pass in range(max_passes):
        improved = False
        for v in range(n):
            if gain[v] > 1e-9:
                # Flip v
                s[v] = -s[v]
                labels[v] = 1 - labels[v]
                improved = True
                # Update As for all neighbors of v
                # As[u] -= 2 * A[v,u] * s[v] for each u (since s[v] changed by 2*old_s[v] -> -2*old_s[v], delta = -2*old_s[v] = 2*s[v]_new... let me be careful)
                # Old s[v] = -s[v]_new. Delta in s[v] = s[v]_new - s[v]_old = -2*s[v]_old = 2*s[v]_new
                # As[u] = sum_w A[u,w] s[w]. The term with w=v: A[u,v] * s[v].
                # Delta As[u] = A[u,v] * delta_s[v] = A[u,v] * 2 * s[v]_new
                # But wait, I already updated s[v]. So s[v]_new = s[v] (current).
                # delta_s = s[v]_new - s[v]_old = -s[v]_old - s[v]_old... 
                # s[v]_old = -s[v]_new. delta = s[v]_new - (-s[v]_new) = 2*s[v]_new.
                # So As[u] += A[u,v] * 2 * s[v] for all u.
                # Since A is symmetric, A[u,v] = A[v,u].
                delta = 2.0 * s[v]
                As += delta * adj[v]
                # Update gain for affected vertices: gain[u] = 0.5 * s[u] * As[u]
                # Only neighbors of v (and v itself) have changed As.
                # For v itself: gain[v] = 0.5 * s[v] * As[v] (recompute)
                # For neighbors: gain[u] = 0.5 * s[u] * As[u]
                # We can just recompute gain for v and its neighbors.
                # But for simplicity in a pass, we can just recompute gain[v] inline.
                gain[v] = 0.5 * s[v] * As[v]
                # For neighbors, their gain changed but we'll recompute when we visit them? 
                # No, we're iterating sequentially. We need to update their gains.
                # Let's update gains for all neighbors of v.
                neighbors = np.nonzero(adj[v])[0]
                for u in neighbors:
                    gain[u] = 0.5 * s[u] * As[u]
        if not improved:
            break
    
    return labels


def _spectral_init(adj, n):
    """
    Use the Fiedler vector (2nd smallest eigenvector of Laplacian) to initialize.
    """
    # Build Laplacian L = D - A
    degree = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(degree)
    A_sparse = sp.csr_matrix(adj)
    L = D - A_sparse
    
    # Find 2nd smallest eigenvalue/eigenvector
    # Use eigsh to find eigenvalues near the smallest
    try:
        # We want the 2nd smallest. Use sigma=0 with which='LM' won't work directly.
        # Use which='SM' to find smallest eigenvalues.
        k = min(4, n - 1)  # get a few smallest
        if k < 2:
            # Fallback: alternating
            return np.arange(n) % 2
        evals, evecs = spla.eigsh(L, k=k, which='SM')
        # evecs columns are eigenvectors sorted by eigenvalue
        # The 2nd smallest is index 1 (index 0 is the trivial all-ones vector)
        fiedler = evecs[:, 1]
        # Sign-based partition
        labels = (fiedler > 0).astype(int)
        return labels
    except Exception:
        # Fallback
        return np.arange(n) % 2


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Greedy Local Search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry (in case of floating point issues)
    adj = 0.5 * (adj + adj.T)
    
    # Spectral initialization
    labels = _spectral_init(adj, n)
    labels = _greedy_refine(adj, labels)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, best_labels)
    
    # Multi-start: try a few random inits if time permits
    rng = np.random.default_rng(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        # Random init
        labels_r = rng.integers(0, 2, size=n)
        labels_r = _greedy_refine(adj, labels_r)
        cut_r = _cut_size(adj, labels_r)
        if cut_r > best_cut:
            best_cut = cut_r
            best_labels = labels_r
    
    # Final polish: one more greedy pass on the best
    if time.time() < deadline:
        polished = _greedy_refine(adj, best_labels)
        polished_cut = _cut_size(adj, polished)
        if polished_cut > best_cut:
            best_labels = polished
            best_cut = polished_cut
    
    return best_labels
# EVOLVE-BLOCK-END
