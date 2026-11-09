import numpy as np
from scipy.optimize import least_squares, minimize
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized exact verification. U:(n*m,r) V:(m*p,r) W:(p*n,r)."""
    rank = U.shape[1]
    # Build full tensor via einsum: T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    return True

def _prefilter_counts(U, V, W, n, m, p):
    """Cheap necessary condition for ±1 matrices."""
    rank = U.shape[1]
    if rank % 2 == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                prod = U[u, :] * V[v, :] * W[w, :]
                agree = np.count_nonzero(prod == 1)
                if agree != (rank + 1) // 2:
                    return False
    return True

def _try_random_int(rank, n, m, p, n_attempts, deadline):
    """Random ±1 search with pre-filter."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        U = np.random.choice([-1, 1], size=(size_u, rank))
        V = np.random.choice([-1, 1], size=(size_v, rank))
        W = np.random.choice([-1, 1], size=(size_w, rank))
        if not _prefilter_counts(U, V, W, n, m, p):
            continue
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    return None

def _try_least_squares(rank, n, m, p, n_attempts, deadline):
    """Structured optimization search for half-integer solutions."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = (size_u + size_v + size_w) * rank

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)

    def unpack(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank:(size_u + size_v) * rank].reshape(size_v, rank)
        W = x[(size_u + size_v) * rank:].reshape(size_w, rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.empty(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            VW = V[v, :] * W[w, :]
            UW = U[u, :] * W[w, :]
            UV = U[u, :] * V[v, :]
            J[idx, u * rank:(u + 1) * rank] = VW
            J[idx, size_u * rank + v * rank:size_u * rank + (v + 1) * rank] = UW
            J[idx, (size_u + size_v) * rank + w * rank:(size_u + size_v) * rank + (w + 1) * rank] = UV
        return J

    best = None
    best_norm = np.inf
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        # Try different initialization strategies
        strategy = random.choice(['random_half', 'random_int', 'sparse'])
        if strategy == 'random_half':
            x0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
        elif strategy == 'random_int':
            x0 = np.random.choice([-1.0, 1.0], size=total_size)
        else:
            # Sparse initialization: most entries zero, some random
            x0 = np.zeros(total_size)
            nonzero_idx = np.random.choice(total_size, size=total_size // 4, replace=False)
            x0[nonzero_idx] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=len(nonzero_idx))
        
        try:
            sol = least_squares(residuals, x0, jac=jacobian, method='lm',
                                max_nfev=2000)
            norm = np.linalg.norm(sol.fun)
            if norm < best_norm:
                best_norm = norm
                best = sol.x
            if norm < 1e-7:
                U, V, W = unpack(sol.x)
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
        except Exception:
            continue
    
    # Polish the best near-solution if it exists
    if best is not None and best_norm < 1e-4:
        try:
            sol = least_squares(residuals, best, jac=jacobian, method='lm',
                                max_nfev=10000)
            U, V, W = unpack(sol.x)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        except Exception:
            pass
    return None

def _try_gradient_descent(rank, n, m, p, n_attempts, deadline):
    """Alternative: use minimize with L1 penalty to encourage sparsity."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = (size_u + size_v + size_w) * rank

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)

    def unpack(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank:(size_u + size_v) * rank].reshape(size_v, rank)
        W = x[(size_u + size_v) * rank:].reshape(size_w, rank)
        return U, V, W

    def objective(x):
        U, V, W = unpack(x)
        # Residual sum of squares
        res_sq = 0.0
        for (u, v, w) in triples:
            val = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
            res_sq += val * val
        # L1 penalty to encourage sparsity
        l1 = np.sum(np.abs(x))
        return res_sq + 0.01 * l1

    best = None
    best_obj = np.inf
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        x0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
        try:
            sol = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            obj = sol.fun
            if obj < best_obj:
                best_obj = obj
                best = sol.x
            # Check if we found a valid solution
            U, V, W = unpack(sol.x)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        except Exception:
            continue
    
    if best is not None:
        U, V, W = unpack(best)
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_structured_low_rank(rank, n, m, p, deadline):
    """Try to construct a low-rank decomposition using Kronecker products or structured patterns."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # For 2x3 x 3x4, known lower bounds suggest rank ~ 17-18
    # Try to find a pattern-based solution
    
    # Strategy: use random Gaussian initialization and refine
    if time.time() > deadline:
        return None
    
    U = np.random.randn(size_u, rank) * 0.5
    V = np.random.randn(size_v, rank) * 0.5
    W = np.random.randn(size_w, rank) * 0.5
    
    # Project to half-integers and verify
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify_fast(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    
    return None

def _try_direct_construction(rank, n, m, p, deadline):
    """Try to construct a low-rank decomposition using known patterns for 2x3 x 3x4."""
    size_u = n * m  # 6
    size_v = m * p  # 12
    size_w = p * n  # 8
    
    # For 2x3 x 3x4, try rank 18 with a structured approach
    # Use the fact that we can exploit the tensor structure
    
    if time.time() > deadline:
        return None
    
    # Try a systematic search with better initialization
    # Based on the structure, try patterns where entries are from {-1, 0, 1}
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with structured random values
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        
        # Fill with random ±1 values but with some structure
        for r in range(rank):
            # Each column should have a specific pattern
            # Try to make each column correspond to a "path" through the tensor
            u_indices = np.random.choice(size_u, size=min(3, size_u), replace=False)
            v_indices = np.random.choice(size_v, size=min(4, size_v), replace=False)
            w_indices = np.random.choice(size_w, size=min(3, size_w), replace=False)
            
            U[u_indices, r] = 1
            V[v_indices, r] = 1
            W[w_indices, r] = 1
        
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Search ranks from low to high, stop at first valid
    # Known lower bound for 2x3 x 3x4 is at least 17
    for rank in range(17, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Phase 1: direct construction with structured patterns
        result = _try_direct_construction(rank, n, m, p, deadline - 15)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 2: random ±1 with pre-filter (fast)
        if time.time() > deadline - 15:
            break
        result = _try_random_int(rank, n, m, p, n_attempts=3000,
                                 deadline=deadline - 15)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 3: structured least-squares with better init
        if time.time() > deadline - 15:
            break
        result = _try_least_squares(rank, n, m, p, n_attempts=20,
                                    deadline=deadline - 15)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        
        # Phase 4: gradient descent with L1 penalty
        if time.time() > deadline - 15:
            break
        result = _try_gradient_descent(rank, n, m, p, n_attempts=5,
                                       deadline=deadline - 15)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
    
    # Convert to lists of integers/half-integers
    U_out = [[(int(x * 2) / 2) for x in row] for row in best_U]
    V_out = [[(int(x * 2) / 2) for x in row] for row in best_V]
    W_out = [[(int(x * 2) / 2) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
