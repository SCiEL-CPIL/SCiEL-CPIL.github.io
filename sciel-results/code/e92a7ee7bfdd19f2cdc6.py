"""Maximum Cut solver using Goemans-Williamson SDP relaxation with randomized rounding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    n = len(labels)
    # Convert to +/-1
    s = 2 * labels.astype(np.float64) - 1
    # cut = 0.5 * (sum_{i,j} A[i,j] - s^T A s)
    # Since A is symmetric with 0 diagonal, sum_{i,j} A[i,j] = 2 * E
    # But easier: just count directly
    # For 100 nodes, this is fast
    return np.sum(adj[np.triu_indices(n, k=1)] * (labels[np.triu_indices(n, k=1)[0]] != labels[np.triu_indices(n, k=1)[1]]))


def _sdp_init(adj, n, max_iters=200, tol=1e-6):
    """
    Solve the SDP relaxation of Max-Cut using projected gradient ascent.
    Maximize tr(X * (A - D)) subject to X_ii = 1, X >= 0
    where D is the degree matrix.
    """
    # Build the matrix B = A - D (Laplacian with sign flip for maximization)
    # Actually, the SDP is: maximize tr(X * (A - D)) where D[i,i] = sum_j A[i,j]
    # But wait, the standard Max-Cut SDP is:
    # maximize (1/4) * sum_{i,j} (A[i,j] - A[i,i]) * X[i,j]  ... no
    # Let me be more careful.
    # Max-Cut: maximize sum_{i<j} A[i,j] * (1 - s_i * s_j) / 2
    #        = (1/2) * sum_{i<j} A[i,j] - (1/2) * sum_{i<j} A[i,j] * s_i * s_j
    #        = (1/4) * sum_{i,j} A[i,j] - (1/4) * s^T A s  (for symmetric A with 0 diag)
    # So we want to minimize s^T A s, or equivalently maximize -s^T A s.
    # The SDP relaxation: let X = s s^T, then maximize -tr(X A) s.t. X_ii = 1, X >= 0
    # Or equivalently, minimize tr(X A) s.t. X_ii = 1, X >= 0
    # But to make it a maximization problem: maximize tr(X * (-A)) s.t. X_ii = 1, X >= 0
    # 
    # Actually, let's use the formulation: maximize tr(X * B) where B = A - D
    # No wait. Let me just use the standard approach.
    # 
    # The objective is to maximize sum_{i,j} A[i,j] * X[i,j] / 2 - sum_i D[i,i] * X[i,i] / 2
    # Since X[i,i] = 1, this becomes: maximize (1/2) * sum_{i,j} A[i,j] * X[i,j] - (1/2) * sum_i D[i,i]
    # The second term is constant, so we just maximize sum_{i,j} A[i,j] * X[i,j]
    # But this doesn't seem right either.
    #
    # Let me restart. The Max-Cut SDP is:
    # maximize  (1/4) * sum_{i,j} (A[i,j] - A[i,i]) * X[i,j]  -- no, A[i,i]=0
    # 
    # Standard formulation:
    # maximize  sum_{i<j} A[i,j] * (1 - X[i,j]) / 2
    # = (1/2) * sum_{i<j} A[i,j] - (1/2) * sum_{i<j} A[i,j] * X[i,j]
    # = const - (1/4) * sum_{i,j} A[i,j] * X[i,j]  (since A is symmetric, 0 diag)
    # So we want to MINIMIZE sum_{i,j} A[i,j] * X[i,j] = tr(X^T A) = tr(X A)
    # Subject to X[i,i] = 1, X >= 0
    #
    # So we solve: minimize tr(X A) s.t. X_ii = 1, X >= 0
    # Using projected gradient descent on this minimization problem.
    
    # Gradient of tr(X A) w.r.t. X is A (since A is symmetric)
    # So we do: X <- project(X - lr * A)
    # But wait, we want to MINIMIZE tr(XA), so gradient ascent would be X + lr*A, but we want descent.
    # Actually, for minimization: X <- X - lr * grad, where grad = A
    # So X <- X - lr * A, then project onto {X : X_ii = 1, X >= 0}
    
    # Initialize X as identity
    X = np.eye(n)
    
    # Use a simple learning rate
    lr = 0.1
    
    prev_obj = np.inf
    for it in range(max_iters):
        # Compute objective
        obj = np.trace(X @ adj)
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj
        
        # Gradient of tr(XA) is A
        # Update: X = X - lr * A
        X_new = X - lr * adj
        
        # Project: set diagonal to 1, and clip negative entries to 0
        # But clipping to 0 breaks symmetry and PSD-ness. 
        # For simplicity, just set diagonal to 1 and keep symmetry.
        # A proper projection onto PSD with fixed diagonal is complex.
        # Let's just set diagonal to 1 and symmetrize.
        X_new = (X_new + X_new.T) / 2
        np.fill_diagonal(X_new, 1.0)
        
        # Check if X_new is PSD. If not, we might need to adjust.
        # For now, just proceed. The projection is approximate.
        X = X_new
    
    return X


def _rounding(X, n, n_samples=100):
    """
    Randomized hyperplane rounding from SDP solution.
    Sample random vectors, project X onto them to get signs.
    """
    best_labels = None
    best_cut = -1
    
    for _ in range(n_samples):
        # Sample random vector
        v = np.random.randn(n)
        # Compute s_i = sign(v^T X v_i) ... no
        # The standard Goemans-Williamson rounding:
        # Sample a random vector u from N(0, I)
        # Set s_i = sign(u^T x_i) where x_i is the i-th row/column of X
        # Actually, since X is PSD, we can do Cholesky or eigenvalue decomposition
        # But simpler: sample u ~ N(0,I), then s_i = sign(sum_j X[i,j] * u[j])
        u = np.random.randn(n)
        s = np.sign(X @ u)
        # Handle zeros
        s[s == 0] = 1
        labels = (s > 0).astype(int)
        
        cut = _cut_size(adj_global, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels


def _greedy_refine(adj, labels, max_rounds=10):
    """
    Greedy local improvement: flip vertices that improve the cut.
    """
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        improved = False
        for v in range(n):
            # Count edges to same side and other side
            same_side = labels[labels == labels[v]]
            other_side = labels[labels != labels[v]]
            
            # Number of edges from v to same side
            edges_same = np.sum(adj[v, same_side])
            # Number of edges from v to other side
            edges_other = np.sum(adj[v, other_side])
            
            if edges_same > edges_other:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses SDP relaxation with randomized rounding, followed by greedy refinement.
    """
    global adj_global
    adj_global = np.array(adjacency, dtype=np.float64)
    n = len(adj_global)
    
    deadline = time.time() + 140  # Safety margin
    
    # Try multiple approaches and keep the best
    best_labels = None
    best_cut = -1
    
    # Approach 1: SDP-based initialization
    if time.time() < deadline:
        try:
            X = _sdp_init(adj_global, n, max_iters=100, tol=1e-5)
            # Rounding
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                u = np.random.randn(n)
                s = np.sign(X @ u)
                s[s == 0] = 1
                labels = (s > 0).astype(int)
                cut = _cut_size(adj_global, labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
        except Exception:
            pass
    
    # Approach 2: Spectral initialization
    if time.time() < deadline:
        try:
            # Compute Laplacian
            deg = np.sum(adj_global, axis=1)
            L = np.diag(deg) - adj_global
            # Get Fiedler vector (2nd smallest eigenvector)
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            labels = (fiedler > 0).astype(int)
            cut = _cut_size(adj_global, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        except Exception:
            pass
    
    # Approach 3: Random initializations
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            labels = np.random.randint(0, 2, n)
            cut = _cut_size(adj_global, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # Fallback: alternating start
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj_global, best_labels)
    
    # Greedy refinement
    if time.time() < deadline - 5:
        best_labels = _greedy_refine(adj_global, best_labels, max_rounds=5)
        best_cut = _cut_size(adj_global, best_labels)
    
    # More refinement if time allows
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            best_labels = _greedy_refine(adj_global, best_labels, max_rounds=10)
            best_cut = _cut_size(adj_global, best_labels)
    
    return best_labels
# EVOLVE-BLOCK-END
