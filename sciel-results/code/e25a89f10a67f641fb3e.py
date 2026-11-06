# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # ±1
    # s^T A s counts 2 for each same-side edge and -2 for each crossing edge? No.
    # Let's use the direct formula:
    # cut = sum_{i,j} A[i,j] * (labels[i] != labels[j])
    # In terms of s: labels[i] != labels[j] iff s[i]*s[j] == -1
    # (1 - s[i]*s[j])/2 is 1 if different, 0 if same
    # cut = 0.5 * sum_{i,j} A[i,j] * (1 - s[i]*s[j])
    #     = 0.5 * (sum A - s^T A s)
    # But sum A is 2*E (since undirected, A is symmetric with 0 diagonal)
    # Actually, let's just compute it directly to be safe and fast with numpy
    # For 100 nodes, O(n^2) is 10,000 ops, which is trivial.
    
    # Efficient vectorized way:
    # diff = labels[:, None] != labels[None, :]
    # return np.sum(adj * diff)
    
    # Even faster using the s formulation:
    # s^T A s = sum_i sum_j s_i A_ij s_j
    # For undirected graph with A symmetric:
    # s^T A s = sum_{i<j} 2 * A_ij * s_i s_j
    # If s_i s_j = 1 (same), contributes 2 A_ij
    # If s_i s_j = -1 (diff), contributes -2 A_ij
    # Let S = sum of same-side edges (counted once), D = sum of diff-side edges (counted once)
    # s^T A s = 2S - 2D
    # Also 2E = 2S + 2D => S + D = E
    # So 2S - 2D = 2S - 2(E - S) = 4S - 2E
    # => S = (s^T A s + 2E) / 4
    # D = E - S = E - (s^T A s + 2E)/4 = (4E - s^T A s - 2E)/4 = (2E - s^T A s)/4
    # Cut size is D.
    
    # Let's just do the direct boolean mask approach, it's clear and fast enough for n=100
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff)


def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    # Build Laplacian L = D - A
    # For undirected graph
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    # Find the second smallest eigenvector of L
    # Use scipy sparse eigensolver for efficiency
    L_sparse = sp.csr_matrix(L)
    
    try:
        # We want the Fiedler vector (eigenvector corresponding to 2nd smallest eigenvalue)
        # sigma='LM' finds largest magnitude. We can shift.
        # Or just use dense eig for n=100, it's fast.
        evals, evecs = np.linalg.eigh(L)
        # evecs[:, 1] is the Fiedler vector
        fiedler = evecs[:, 1]
    except:
        # Fallback to random
        return np.random.randint(0, 2, n).astype(np.float64)
    
    # Use sign of Fiedler vector as initial labels
    labels = (fiedler > 0).astype(np.float64)
    
    # Ensure both classes are non-empty
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return labels


def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search.
    Repeatedly find the vertex whose flip increases the cut the most.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    def current_cut():
        s = 2 * labels - 1
        # cut = 0.5 * (sum_adj - s^T A s)
        # But sum_adj is constant. Let's just compute directly for clarity in optimization loop
        # Actually, let's use the s formulation for speed
        return 0.5 * (np.sum(adj) - s @ adj @ s)
    
    best_cut = current_cut()
    
    while True:
        if time.time() > deadline:
            break
        
        # For each vertex v, compute the gain from flipping v
        # Gain(v) = (new_cut - old_cut)
        # new_cut = old_cut + sum_{u in same(v)} A[v,u] - sum_{u in diff(v)} A[v,u]
        # Wait, flipping v changes edges (v,u):
        # If u was same, now diff: gain +A[v,u]
        # If u was diff, now same: gain -A[v,u]
        # So Gain(v) = sum_{u same} A[v,u] - sum_{u diff} A[v,u]
        
        s = 2 * labels - 1  # ±1
        # same neighbors: s[u] == s[v] => s[u]*s[v] == 1
        # diff neighbors: s[u] == -s[v] => s[u]*s[v] == -1
        
        # For vertex v:
        # sum_same(v) = sum_u A[v,u] * (1 + s[v]*s[u]) / 2
        # sum_diff(v) = sum_u A[v,u] * (1 - s[v]*s[u]) / 2
        # Gain(v) = sum_same(v) - sum_diff(v)
        #         = sum_u A[v,u] * (1 + s[v]*s[u])/2 - sum_u A[v,u] * (1 - s[v]*s[u])/2
        #         = sum_u A[v,u] * ( (1 + s[v]*s[u]) - (1 - s[v]*s[u]) ) / 2
        #         = sum_u A[v,u] * s[v]*s[u]
        #         = s[v] * sum_u A[v,u] * s[u]
        #         = s[v] * (A @ s)[v]
        
        As = adj @ s
        gains = s * As
        
        # Find vertex with max positive gain
        max_gain = np.max(gains)
        
        if max_gain <= 0:
            break  # Local optimum
        
        # Flip the vertex with max gain
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        s[v] = -s[v]
        
        # Update best_cut
        best_cut += max_gain
        
        # Optional: if gain is very small, we might be stuck in a plateau
        # But since we always take the best move, we should converge
    
    return labels, best_cut


def _random_init(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, n).astype(np.float64)


def partition(adjacency):
    """
    Find a high-quality cut using spectral initialization and vectorized
    best-improvement local search with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    # Deadline: leave 10s safety margin before the 120s limit (or 150s if that's the kill)
    # The prompt says 120s for evaluation, 150s kill. Let's use 110s to be safe.
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Strategy:
    # 1. Spectral init + refine
    # 2. Multiple random restarts + refine
    # Keep the best result found
    
    # First try spectral initialization
    spec_labels = _spectral_init(adj, n)
    refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels
    
    # Now do random restarts
    # How many can we do? Each refine is O(n^2 * iterations).
    # For n=100, n^2 = 10,000. If iterations ~ 100, that's 1M ops.
    # Should be fast. Let's do as many as we can within the deadline.
    
    rng = np.random.RandomState(42)  # For reproducibility in restarts
    
    restart_count = 0
    while time.time() < deadline - 2.0:  # Leave 2s margin
        random_labels = _random_init(n)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels
        
        restart_count += 1
        
        # If we've done many restarts, we might be stuck
        # We could try different strategies, but for now, keep going
        
        # Safety: if we've done > 1000 restarts, stop
        if restart_count > 2000:
            break
    
    # Final polish: do a few more passes on the best solution
    # The greedy refine should already be at a local optimum
    # But let's do one more check with a slightly different criterion
    
    if best_labels is None:
        best_labels = _random_init(n)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
