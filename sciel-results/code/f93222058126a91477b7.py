"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_step(U, V, W, T, n, m, p, rank):
    """
    Perform one step of ALS for CP decomposition.
    Returns updated U, V, W.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Pre-compute norms for efficiency?
    # Update U
    for r in range(rank):
        # Residual
        # T - sum_{k!=r} U[:,k] outer V[:,k] outer W[:,k]
        # Efficiently: T_approx = einsum(U, V, W). 
        # Residual = T - T_approx + outer(U[:,r], V[:,r]) outer W[:,r]
        # Actually, computing full T_approx every step is O(rank * dim^3).
        # For 3x3x3, dim=9, rank=23. 23 * 729 * 3 ops ~ 50k ops. Very fast.
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # Add back the r-th term to get residual excluding r
        # Residual = T - (T_approx - u_r v_r w_r) = T - T_approx + u_r v_r w_r
        # Wait, T_approx includes term r.
        # R = T - sum_{k!=r} u_k v_k w_k = T - (T_approx - u_r v_r w_r)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # Update U[:, r]
        # Minimize || R - u v w ||^2 w.r.t u
        # b_a = sum_{b,c} R_{abc} V_{rb} W_{rc}
        # A = sum_{b,c} V_{rb}^2 W_{rc}^2
        b = np.einsum('abc,b,c->a', R, V[:,r], W[:,r])
        A = np.sum(V[:,r]**2 * W[:,r]**2)
        if A > 1e-12:
            U[:,r] = b / A
        else:
            U[:,r] = 0.0
            
    # Update V
    for r in range(rank):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        b = np.einsum('abc,a,c->b', R, U[:,r], W[:,r])
        A = np.sum(U[:,r]**2 * W[:,r]**2)
        if A > 1e-12:
            V[:,r] = b / A
        else:
            V[:,r] = 0.0
            
    # Update W
    for r in range(rank):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_approx + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        b = np.einsum('abc,a,b->c', R, U[:,r], V[:,r])
        A = np.sum(U[:,r]**2 * V[:,r]**2)
        if A > 1e-12:
            W[:,r] = b / A
        else:
            W[:,r] = 0.0
            
    return U, V, W

def _try_rank_optimization(target_rank, n, m, p, deadline, max_attempts=50):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    rng = random.Random(42)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialization Strategy:
        # 1. Pick target_rank columns from schoolbook
        # 2. Add small noise
        # 3. Or random init
        
        if attempt < 20:
            # Schoolbook subset
            cols = rng.sample(range(27), target_rank)
            U_init = U0[:, cols].copy()
            V_init = V0[:, cols].copy()
            W_init = W0[:, cols].copy()
            # Perturb to escape local minima
            U_init += np.random.randn(dim_a, target_rank) * 0.01
            V_init += np.random.randn(dim_b, target_rank) * 0.01
            W_init += np.random.randn(dim_c, target_rank) * 0.01
        else:
            # Random init
            scale = 0.5
            U_init = np.random.randn(dim_a, target_rank) * scale
            V_init = np.random.randn(dim_b, target_rank) * scale
            W_init = np.random.randn(dim_c, target_rank) * scale

        U, V, W = U_init, V_init, W_init
        
        # Run ALS
        for it in range(200):
            if time.time() > deadline:
                break
            U, V, W = _als_step(U, V, W, T_exp, n, m, p, target_rank)
            
            # Check error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_approx - T_exp)**2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-10:
                return U, V, W
                
    if best_error < 1e-7:
        return best_U, best_V, best_W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try rank 23
    if time.time() < deadline:
        result = _try_rank_optimization(23, n, m, p, deadline, max_attempts=100)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
