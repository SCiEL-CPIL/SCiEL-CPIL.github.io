"""
Solver for QAPLIB wil100 using Burer-Monteiro style SDP relaxation 
with Gradient Descent on the doubly-stochastic manifold, followed by 
Hungarian rounding and local search.
"""
# EVOLVE-BLOCK-START
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
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _project_doubly_stochastic(X, a, b, steps=10):
    """
    Project X onto the set of doubly stochastic matrices using Sinkhorn-Knopp.
    Weights are adjusted to bias towards lower cost interactions.
    """
    n = X.shape[0]
    # Ensure non-negativity
    X = np.maximum(X, 1e-12)
    
    # Create a weight matrix that encourages low-cost assignments
    # We want to maximize sum X[i,j] * (-cost_proxy)
    # A simple proxy is the row/col sums of a and b
    # But a better proxy for the gradient step is the current linearization:
    # C[i,j] = sum_k a[i,k] B[k,j] + sum_k A[i,k] b[k,j]
    # However, for projection, we just need to maintain DS property.
    # We will use standard Sinkhorn but with a temperature scaling if needed.
    
    for _ in range(steps):
        # Row normalization
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums
        
        # Column normalization
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        X = X / col_sums
        
    return X

def _sdp_relaxation_search(a, b, time_limit):
    """
    Perform gradient descent on the doubly stochastic matrix X.
    Objective: Minimize Tr(A X B X^T) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    This is equivalent to sum_{i,j} A[i,j] (B X^T X B^T)[i,j] ? No.
    
    Let's derive the gradient properly.
    Cost(X) = sum_{i,j} A[i,j] sum_{k,l} X[i,k] B[k,l] X[j,l]
             = Tr(A X B X^T)
    
    Gradient w.r.t X:
    d/dX Tr(A X B X^T) = A^T X B^T + A X B
    (Using d Tr(M X N X^T) = M^T X N^T + M X N)
    
    We minimize this gradient on the DS manifold.
    """
    n = len(a)
    # Initialize X as uniform doubly stochastic matrix
    X = np.ones((n, n)) / n
    
    # Precompute B^T for speed
    b_t = b.T
    a_t = a.T
    
    # Learning rate schedule
    lr_init = 0.05
    lr_min = 0.001
    
    start_time = time.time()
    best_cost = np.inf
    best_p = None
    
    # We will run for a fixed number of iterations or until time limit
    # Since n=100, matrix mult is fast.
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute Gradient
        # G = A^T X B^T + A X B
        # Note: A and B are symmetric in QAP usually, but let's be general.
        AX = a @ X
        AXB = AX @ b
        AXT = a_t @ X
        AXTBT = AXT @ b_t
        
        grad = AXB + AXTBT
        
        # Update X
        lr = lr_init * (lr_min / lr_init) ** (it / max_iters)
        X = X - lr * grad
        
        # Project to DS
        X = _project_doubly_stochastic(X, a, b, steps=5)
        
        # Periodically round to permutation to check cost
        if it % 10 == 0:
            # Hungarian algorithm on -X (maximize similarity)
            # X[i,j] is prob i->j. We want to find p such that sum X[i, p[i]] is max.
            # linear_sum_assignment minimizes cost, so we pass -X.
            row_ind, col_ind = linear_sum_assignment(-X)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            
            cost = _qap_cost(a, b, p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def _local_search(a, b, p, time_limit):
    """First-improvement local search with best-improvement option for tighter polish."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Find best improving swap
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        
        if not improved:
            break
            
    return best_p, best_cost

def _sa_polish(a, b, p, time_limit):
    """Simulated annealing polish phase."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    t0 = 10000.0
    t_final = 0.01
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
                
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: SDP Relaxation Search (Structural Change)
    # Allocate 40% of time to this
    sdp_time = min(45.0, deadline * 0.4)
    p_sdp, cost_sdp = _sdp_relaxation_search(a, b, sdp_time)
    
    if p_sdp is not None and cost_sdp < best_cost:
        best_cost = cost_sdp
        best_p = p_sdp.copy()
        
    # Fallback / Alternative Init: Hungarian on row sums
    # If SDP failed or is slow, we have a backup.
    # Actually, let's also try a standard Hungarian init to have a strong baseline
    # if SDP gets stuck in a bad local min of the relaxation.
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        # Normalize
        ars_max = np.max(np.abs(a_row_sums))
        brs_max = np.max(np.abs(b_row_sums))
        if ars_max > 0: ars_norm = a_row_sums / ars_max
        else: ars_norm = a_row_sums
        if brs_max > 0: brs_norm = b_row_sums / brs_max
        else: brs_norm = b_row_sums
        
        C = -(ars_norm[:, None] * brs_norm[None, :])
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        cost_hung = _qap_cost(a, b, p_hung)
        
        if cost_hung < best_cost:
            best_cost = cost_hung
            best_p = p_hung.copy()
            
    # Phase 2: Local Search Polish
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5 and best_p is not None:
        ls_time = min(20.0, time_remaining * 0.3)
        p_ls, cost_ls = _local_search(a, b, best_p.copy(), ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # Phase 3: SA Polish on remaining time
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5 and best_p is not None:
        sa_time = min(30.0, time_remaining * 0.6)
        p_sa, cost_sa = _sa_polish(a, b, best_p.copy(), sa_time)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
            
    # Final check
    if best_p is None:
        # Absolute fallback: identity
        best_p = np.arange(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
