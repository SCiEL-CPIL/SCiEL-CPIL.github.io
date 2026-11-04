"""
Continuous Relax-and-Round Solver for QAPLIB wil100.
Strategy:
1. Construct a linear surrogate cost matrix C from QAP structure (row/col sums interaction).
2. Optimize a doubly-stochastic matrix X (rows=facilities, cols=locations) via Sinkhorn-regularized
   gradient descent to minimize trace(X^T C X) approximately (linearized as <X, C>).
   This finds a 'soft' assignment that respects the global structure better than Hungarian alone.
3. Round the soft assignment to a hard permutation using the Hungarian algorithm on the final soft costs.
4. Polish the resulting permutation with a fast first-improvement 2-opt local search.
5. If time remains, perform a second round of soft optimization starting from the polished permutation's
   one-hot matrix, then round and polish again.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _build_surrogate(a, b):
    """
    Build a surrogate cost matrix for the linear assignment problem.
    Uses a combination of row-sum and column-sum interactions to capture
    the quadratic structure in a linear form.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Primary surrogate: align high flow with high distance
    C1 = -np.outer(a_row_sums, b_row_sums)
    C2 = -np.outer(a_col_sums, b_col_sums)
    
    # Combine for robustness
    C = C1 + C2
    return C

def _sinkhorn_round(C, n, max_iter=50, eps=1e-3):
    """
    Perform Sinkhorn iterations to find a doubly-stochastic matrix X
    that approximately minimizes <X, C>.
    Returns the rounded permutation.
    """
    # Initialize with uniform matrix
    X = np.ones((n, n)) / n
    
    # Sinkhorn scaling to minimize <X, C>
    # Gradient of <X, C> w.r.t. log X is C
    # Update: X = exp(-C / eps) * scaling
    # We use a simple iterative scaling approach
    
    # Start with K = exp(-C / eps)
    K = np.exp(-C / eps)
    
    # Normalize to doubly stochastic
    for _ in range(max_iter):
        # Row scaling
        row_sums = K.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        K = K / row_sums
        
        # Column scaling
        col_sums = K.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        K = K / col_sums
    
    # Round using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(K)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _soft_refine(a, b, p_init, max_iter=100, eps=1e-2):
    """
    Refine a permutation using soft assignment optimization.
    Starts from the one-hot matrix of p_init and performs gradient steps
    on the doubly-stochastic manifold.
    """
    n = len(p_init)
    
    # Build surrogate
    C = _build_surrogate(a, b)
    
    # Initialize X as one-hot from p_init
    X = np.zeros((n, n))
    X[np.arange(n), p_init] = 1.0
    
    # Perform Sinkhorn-like refinement
    # We'll do a few rounds of:
    # 1. Compute gradient g = C (for linear objective <X, C>)
    # 2. Update log X -= lr * g
    # 3. Renormalize to doubly stochastic
    
    lr = 0.1
    for it in range(max_iter):
        # Gradient of <X, C> is C
        # Update in log space: log_X -= lr * C
        log_X = np.log(X + 1e-10)
        log_X -= lr * C
        X = np.exp(log_X)
        
        # Renormalize to doubly stochastic
        for _ in range(5):  # A few Sinkhorn iterations
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            X = X / col_sums
    
    # Round
    row_ind, col_ind = linear_sum_assignment(X)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    return p_new

def _local_search(a, b, p, max_iters=200):
    """
    First-improvement local search using 2-opt swaps.
    Returns improved permutation and cost.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
        if not improved:
            break
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Phase 1: Build surrogate and get initial soft assignment
    C = _build_surrogate(a, b)
    
    # Use Hungarian on surrogate for initial hard assignment
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Phase 2: Soft refinement from initial assignment
    if time.time() - start_time < total_deadline - 30:
        p_refined = _soft_refine(a, b, p, max_iter=50, eps=1e-2)
        cost_refined = _qap_cost(a, b, p_refined)
        cost_init = _qap_cost(a, b, p)
        if cost_refined < cost_init:
            p = p_refined
            cost_init = cost_refined
    
    # Phase 3: Local search polish
    p, current_cost = _local_search(a, b, p, max_iters=300)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 4: If time remains, do another round of soft refinement
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 15:
        p_soft = _soft_refine(a, b, best_p, max_iter=80, eps=1e-3)
        cost_soft = _qap_cost(a, b, p_soft)
        if cost_soft < best_cost:
            p_ls, cost_ls = _local_search(a, b, p_soft, max_iters=200)
            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
    
    # Phase 5: Final local search polish if time allows
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 5:
        p_ls, cost_ls = _local_search(a, b, best_p, max_iters=100)
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
    
    return best_p.tolist()
