"""
Strategy: Continuous Soft-Assignment QAP Solver.
Reframes the QAP as an optimization over the Birkhoff polytope (doubly-stochastic matrices).
Uses gradient descent with Sinkhorn regularization to find a high-quality soft assignment,
extracts a hard permutation via the Hungarian algorithm, and polishes with fast 2-opt.
This explores a different solution space (continuous) than pure combinatorial search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym=None):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    """
    if a_sym is None:
        a_sym = a + a.T
    
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _sinkhorn_step(X, eps, iterations=50):
    """
    Perform Sinkhorn scaling to make X approximately doubly stochastic.
    """
    n = X.shape[0]
    u = np.ones(n)
    v = np.ones(n)
    
    # Softmax to ensure positivity
    X = np.exp(X / eps)
    
    for _ in range(iterations):
        u = 1.0 / (X @ v)
        v = 1.0 / (X.T @ u)
    
    return np.diag(u) @ X @ np.diag(v)

def _soft_assignment_init(a, b, time_limit, a_sym=None):
    """
    Initialize using gradient descent on the Birkhoff polytope.
    Objective: min trace(A X B X^T) + entropy_regularization
    """
    n = len(a)
    start_time = time.time()
    
    if a_sym is None:
        a_sym = a + a.T
    
    # Start with random permutation
    p0 = np.random.permutation(n)
    X = np.eye(n)[p0]
    
    # Precompute for gradient
    # Gradient of Q(X) = trace(A X B X^T) is 2 * A X B (if A, B symmetric)
    # We use the form: sum a_ij b_p(i)p(j)
    # Let's use the standard QAP formulation on Birkhoff polytope.
    
    # We will minimize F(X) = <A, X B X^T>
    # Using Sinkhorn to project back to Birkhoff polytope after gradient step.
    
    # Initial X
    X = np.random.rand(n, n)
    X = _sinkhorn_step(X, 1.0, 100)
    
    # Gradient descent with Sinkhorn projection
    lr = 1e-3
    eps = 1.0
    momentum = 0.9
    v = np.zeros_like(X)
    
    best_cost = np.inf
    best_p = p0.copy()
    
    for step in range(500):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Gradient: 2 * A @ X @ B (assuming A, B symmetric, which we ensured for a_sym, but b might not be)
        # Actually, cost is sum a_ij b_kl X_ik X_jl
        # Gradient w.r.t X_ik: sum_jl a_ij b_kl X_jl = (A X B)_ik
        # Since we want to minimize, we subtract gradient.
        grad = a_sym @ X @ b
        # Note: if b is not symmetric, we should use (A X B + A X B^T) / 2? 
        # Let's stick to the standard: grad = A X B + A X B.T is for trace(A X B X^T)?
        # Q(X) = trace(A X B X^T) = sum_ijkl A_ik B_jl X_ij X_kl
        # d/dX_ij = sum_kl A_ik B_jl X_kl + sum_kl A_jk B_il X_kl
        # = (A X B)_ji + (A X B.T)_ji
        # Let's just use grad = a_sym @ X @ b + a_sym @ X @ b.T
        grad = a_sym @ X @ b + a_sym @ X @ b.T
        
        # Update
        v = momentum * v + (1 - momentum) * grad
        X = X - lr * v
        
        # Project to Birkhoff polytope using Sinkhorn
        X = _sinkhorn_step(X, eps, 20)
        
        # Extract permutation
        row_ind, col_ind = linear_sum_assignment(-X)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr *= 0.5
            eps *= 0.5
            
    return best_p, best_cost

def _local_search_2opt(a, b, p, time_limit, a_sym=None, max_passes=10):
    """
    Perform 2-opt local search on permutation p.
    Returns the improved permutation and its cost.
    """
    n = len(p)
    if a_sym is None:
        a_sym = a + a.T
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    break
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p, best_cost

def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Strategy:
    # 1. Generate a few diverse soft-assignment inits
    # 2. Polish each with 2-opt
    # 3. Keep the best
    
    best_p = None
    best_cost = np.inf
    
    num_inits = 5
    
    for i in range(num_inits):
        if time.time() - start_time > time_limit - 20.0:
            break
            
        # Soft assignment init
        p_soft, cost_soft = _soft_assignment_init(a, b, time_limit - 15.0, a_sym)
        
        if cost_soft < best_cost:
            best_cost = cost_soft
            best_p = p_soft.copy()
            
        # Polish with 2-opt
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_soft, time_limit - 10.0, a_sym, max_passes=5
        )
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
            
    # Final polish with more time if possible
    if best_p is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 5.0:
            best_p, best_cost = _local_search_2opt(
                a, b, best_p, time_limit - 1.0, a_sym, max_passes=10
            )
            
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
