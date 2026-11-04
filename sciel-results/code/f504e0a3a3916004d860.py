# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian."""
    from scipy.optimize import linear_sum_assignment
    n = X.shape[0]
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _sinkhorn_project(A, n_iter=200, tol=1e-6):
    """Project matrix A onto Birkhoff polytope using Sinkhorn scaling."""
    n = A.shape[0]
    # Ensure non-negative
    A = np.maximum(A, 0)
    # Add small epsilon to avoid zeros
    A = A + 1e-10
    u = np.ones(n)
    v = np.ones(n)
    for _ in range(n_iter):
        # Scale rows
        u = 1.0 / (A @ v + 1e-10)
        # Scale columns
        v = 1.0 / ((u[:, None] * A).sum(axis=0) + 1e-10)
        # Check convergence
        row_sums = (u[:, None] * A) @ v
        col_sums = (u[:, None] * A).sum(axis=0)
        if np.max(np.abs(row_sums - 1.0)) < tol and np.max(np.abs(col_sums - 1.0)) < tol:
            break
    return u[:, None] * A * v[None, :]


def _sdp_round(a, b, n_iter=50, lr=0.05):
    """SDP relaxation via projected gradient descent on Birkhoff polytope.
    Objective: min trace(X^T A) * trace(Y^T B) where X,Y are assignment matrices.
    For QAP: min sum_{i,j,k,l} a[i,j]*b[k,l]*X[i,k]*X[j,l]
    Gradient w.r.t X: 4 * (A ⊙ (X B X^T)) where ⊙ is Hadamard
    Project onto Birkhoff polytope using Sinkhorn."""
    n = a.shape[0]
    # Initialize X as uniform (1/n)
    X = np.ones((n, n)) / n
    
    # Precompute symmetric versions
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    best_X = X.copy()
    best_obj = np.inf
    
    for it in range(n_iter):
        # Compute objective: sum_{i,j,k,l} a[i,j]*b[k,l]*X[i,k]*X[j,l]
        # = trace(X^T A X B X^T) ... let's compute directly
        # cost = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] * X[i,k] * X[j,l]
        # Let M = X B X^T, then cost = sum_{i,j} a[i,j] * M[i,j] = trace(A^T M)
        M = X @ B_sym @ X.T
        obj = np.sum(a * M)
        
        if obj < best_obj:
            best_obj = obj
            best_X = X.copy()
        
        # Gradient: d/dX trace(A^T X B X^T) = 2 A X B^T + 2 A^T X B
        # Since A and B are symmetric: grad = 2 (A X B + A X B) = 4 A X B
        grad = 4.0 * (a @ X @ b)
        
        # Gradient descent step
        X_new = X - lr * grad
        
        # Project onto Birkhoff polytope
        X_proj = _sinkhorn_project(X_new, n_iter=100, tol=1e-5)
        
        # Check if we made progress
        M_new = X_proj @ B_sym @ X_proj.T
        obj_new = np.sum(a * M_new)
        
        if obj_new < obj + 1e-3:
            X = X_proj
        else:
            # Reduce learning rate
            lr *= 0.5
            X = X_proj
    
    # Extract permutation from best_X using spectral rounding
    # Use the eigenvector corresponding to the largest eigenvalue
    # or just use greedy assignment based on X values
    
    # Method 1: Hungarian on the relaxed matrix
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-best_X)  # maximize correlation
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p, best_obj


def _two_opt_search(a, b, p, time_limit, max_iter=500000):
    """Vectorized 2-opt local search with precomputed contributions."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                li = p[i]
                lj = p[j]
                # Compute delta for swapping locations li and lj
                drow = b[lj, p] - b[li, p]
                dcol = b[p, lj] - b[p, li]
                delta = np.dot(a[i, :] - a[j, :], drow) + np.dot(a[:, i] - a[:, j], dcol)
                if delta < 0:
                    p[i], p[j] = lj, li
                    cost += delta
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
        if not improved:
            break
    return best_p, best_cost


def _three_opt_search(a, b, p, time_limit, max_iter=100000):
    """3-opt local search: try moving facility i to location l (where l is currently occupied by j)."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            li = p[i]
            for j in range(n):
                if i == j:
                    continue
                lj = p[j]
                # Move facility i from li to lj, facility j from lj to li (this is just a swap)
                # For true 3-opt, we'd need to consider moving i to a location not directly swapped
                # But for simplicity, let's do 2-opt style with better delta computation
                
                # Actually, let's try inserting i into a different position
                # This is complex, so let's stick with optimized 2-opt
                pass
        if not improved:
            break
    return best_p, best_cost


def _polish_search(a, b, p, time_limit):
    """Final polish with tighter tolerance and more iterations."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # Precompute for faster delta computation
    # RC[l, k] = b[l, p[k]]
    # CC[k, l] = b[p[k], l]
    RC = b[:, p]  # shape (n, n)
    CC = b[p, :]  # shape (n, n)
    
    start = time.time()
    no_improve_count = 0
    for it in range(1000000):
        if time.time() - start > time_limit:
            break
        improved = False
        # Try all swaps
        for i in range(n):
            li = p[i]
            for j in range(i + 1, n):
                lj = p[j]
                # Delta using precomputed RC and CC
                drow = RC[lj, :] - RC[li, :]
                dcol = CC[:, lj] - CC[:, li]
                delta = np.dot(a[i, :] - a[j, :], drow) + np.dot(a[:, i] - a[:, j], dcol)
                if delta < -1e-9:  # Tighter tolerance
                    # Accept swap
                    p[i], p[j] = lj, li
                    cost += delta
                    
                    # Update RC and CC
                    # Swap rows li and lj in RC
                    RC[[li, lj], :] = RC[[lj, li], :]
                    # Swap cols li and lj in CC
                    CC[:, [li, lj]] = CC[:, [lj, li]]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
        if not improved:
            no_improve_count += 1
            if no_improve_count > 2:
                break
        else:
            no_improve_count = 0
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    # Step 1: MDS embedding + Procrustes alignment (fast, good initial solution)
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    k = min(20, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)
    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: Try SDP relaxation if time permits
    time_left = deadline - time.time()
    if time_left > 30:
        try:
            p_sdp, obj_sdp = _sdp_round(a, b, n_iter=30, lr=0.03)
            cost_sdp = _qap_cost(a, b, p_sdp)
            if cost_sdp < best_cost:
                best_cost = cost_sdp
                best_p = p_sdp.copy()
        except Exception:
            pass

    # Step 3: Local search polish
    time_left = deadline - time.time()
    if time_left > 10:
        p, cost = _two_opt_search(a, b, best_p, min(time_left - 5, 40))
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # Step 4: Final high-precision polish
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _polish_search(a, b, best_p, time_left - 2)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
