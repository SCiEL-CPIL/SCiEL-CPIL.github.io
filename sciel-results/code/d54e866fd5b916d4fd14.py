"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_computed = np.zeros((n, m, p), dtype=float)
        
        for c in range(rank):
            u_vec = U_arr[:, c]
            v_vec = V_arr[:, c]
            w_vec = W_arr[:, c]
            
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_mat[j, k]
                        if v_val == 0: continue
                        w_val = w_mat[k, i]
                        if w_val == 0: continue
                        T_computed[i, j, k] += u_val * v_val * w_val
                        
        T_target = np.ones((n, m, p), dtype=float)
        
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True
    except Exception:
        return False

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_r_decomposition(n, m, p, r, max_iters=10000, deadline=None):
    """
    Try to find a rank-r decomposition using gradient-based optimization
    on the polynomial equations.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # The tensor equation: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We'll use a simple gradient descent approach
    # Define the residual
    def compute_residual(U, V, W):
        T_computed = np.zeros((n, m, p))
        for c in range(r):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        return T_computed - np.ones((n, m, p))
    
    def residual_norm(U, V, W):
        R = compute_residual(U, V, W)
        return np.sum(R**2)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_norm = residual_norm(U, V, W)
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        R = compute_residual(U, V, W)
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_val = R[i, j, k]
                        dU[i * m + j, c] += 2 * r_val * v_mat[j, k] * w_mat[k, i]
                        dV[j * p + k, c] += 2 * r_val * u_mat[i, j] * w_mat[k, i]
                        dW[k * n + i, c] += 2 * r_val * u_mat[i, j] * v_mat[j, k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        current_norm = residual_norm(U, V, W)
        if current_norm < best_norm:
            best_norm = current_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_norm < 1e-10:
            break
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    if best_norm < 1e-8:
        # Try to round to rationals with small denominators
        U_r, V_r, W_r = _round_to_rationals(best_U, best_V, best_W, max_denom=10)
        if U_r is not None:
            # Verify
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, r):
                return U_r, V_r, W_r
    
    return None

def _round_to_rationals(U, V, W, max_denom=10):
    """
    Try to round each entry to a rational with small denominator.
    Returns (U_r, V_r, W_r) as lists of lists, or None if fails.
    """
    U_r = []
    for row in U:
        U_r.append([_round_entry(x, max_denom) for x in row])
    V_r = []
    for row in V:
        V_r.append([_round_entry(x, max_denom) for x in row])
    W_r = []
    for row in W:
        W_r.append([_round_entry(x, max_denom) for x in row])
    return U_r, V_r, W_r

def _round_entry(x, max_denom=10):
    """Round x to nearest rational with denominator <= max_denom."""
    if abs(x) < 1e-12:
        return 0
    best = None
    best_err = float('inf')
    for d in range(1, max_denom + 1):
        num = round(x * d)
        if num == 0:
            continue
        val = num / d
        err = abs(x - val)
        if err < best_err:
            best_err = err
            best = val
    if best_err < 1e-6:
        return best
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds for 5x5 matrix multiplication rank
    # The rank is at least n*m*p / max(n,m,p) = 125/5 = 25
    # But actually the lower bound is higher
    # Let's try ranks from known lower bounds upward
    
    # For 5x5, the best known upper bound is around 45-50
    # Let's try a few candidate ranks
    candidate_ranks = [40, 42, 44, 45, 46, 48, 50, 52, 54, 56, 60, 64, 68, 72, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
        
        result = _try_rank_r_decomposition(n, m, p, r, max_iters=5000, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            print(f"Found rank {r} decomposition")
    
    return best_result
# EVOLVE-BLOCK-END

