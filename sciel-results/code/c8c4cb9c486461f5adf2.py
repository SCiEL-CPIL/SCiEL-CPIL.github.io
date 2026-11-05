"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using vectorized operations."""
    rank = U.shape[1]
    max_err = 0.0
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            temp = Ur[i, :] @ Vr  # shape (p,)
            for k in range(p):
                val = temp[k] * Wr[k, i]
                max_err = max(max_err, abs(val))
    # Actually we need sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # Let me redo this properly
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _compute_residuals_fast(U, V, W, n, m, p):
    """Compute residuals vectorized."""
    rank = U.shape[1]
    residuals = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            temp = Ur[i, :] @ Vr  # shape (p,)
            for k in range(p):
                residuals[i, :, k] += temp * Wr[k, i]
    return (residuals - 1.0).flatten()

def _als_refine(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """ALS refinement with alternating linear solves."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m + j, :] = x
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p + k, :] = x
                except:
                    pass
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                try:
                    x, *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n + i, :] = x
                except:
                    pass
        
        # Check convergence
        res = _compute_residuals_fast(U, V, W, n, m, p)
        if np.max(np.abs(res)) < tol:
            break
    
    return U, V, W

def _try_rank_als(n, m, p, rank, deadline, rng, max_restarts=30):
    """Try to find a rank-rank decomposition using random restarts + ALS."""
    best_err = np.inf
    best_U = best_V = best_W = None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.normal(0, 1, (n*m, rank))
        V = rng.normal(0, 1, (m*p, rank))
        W = rng.normal(0, 1, (p*n, rank))
        
        # ALS refinement
        U, V, W = _als_refine(U, V, W, n, m, p, max_iter=100, tol=1e-10)
        
        # Check residual
        res = _compute_residuals_fast(U, V, W, n, m, p)
        max_res = np.max(np.abs(res))
        
        if max_res < best_err:
            best_err = max_res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_res < 1e-9:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _try_rank_optimize(n, m, p, rank, deadline, rng, max_restarts=10):
    """Try to find a rank-rank decomposition using scipy least_squares."""
    from scipy.optimize import least_squares
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    best_err = np.inf
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        params0 = rng.normal(0, 1, size_U + size_V + size_W)
        
        def residuals(params):
            U = params[:size_U].reshape(n*m, rank)
            V = params[size_U:size_U+size_V].reshape(m*p, rank)
            W = params[size_U+size_V:].reshape(p*n, rank)
            return _compute_residuals_fast(U, V, W, n, m, p)
        
        try:
            result = least_squares(residuals, params0, max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            max_res = np.max(np.abs(result.fun))
            
            if max_res < best_err:
                best_err = max_res
                best_params = result.x.copy()
            
            if max_res < 1e-9:
                break
        except:
            continue
    
    if best_params is not None and best_err < 1e-8:
        U = best_params[:size_U].reshape(n*m, rank)
        V = best_params[size_U:size_U+size_V].reshape(m*p, rank)
        W = best_params[size_U+size_V:].reshape(p*n, rank)
        return U, V, W, best_err
    return None, None, None, best_err

def _round_to_half_int(mat):
    """Round matrix entries to nearest half-integer."""
    return np.round(mat * 2) / 2.0

def _check_half_integer(U, V, W, tol=1e-6):
    """Check if all entries are close to half-integers."""
    for mat in [U, V, W]:
        rounded = np.round(mat * 2) / 2.0
        if np.max(np.abs(mat - rounded)) > tol:
            return False
    return True

def _pencil_method_init(n, m, p, rank, rng):
    """
    Use the pencil method idea: construct a structured initialization.
    For 2x3 * 3x4, we know the rank is 19 (from literature).
    We create a structured random initialization that biases towards
    solutions with algebraic structure.
    """
    # Create base patterns
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with random values but with some structure
    for r in range(rank):
        # Random signs and magnitudes
        U[:, r] = rng.choice([-1, 0, 1], size=n*m) * rng.uniform(0.5, 1.5, n*m)
        V[:, r] = rng.choice([-1, 0, 1], size=m*p) * rng.uniform(0.5, 1.5, m*p)
        W[:, r] = rng.choice([-1, 0, 1], size=p*n) * rng.uniform(0.5, 1.5, p*n)
    
    return U, V, W

def _structured_init(n, m, p, rank, rng):
    """
    Structured initialization based on known patterns.
    For 2x3*3x4, use block structure.
    """
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Mix of random and structured
    n_structured = rank // 2
    n_random = rank - n_structured
    
    # Structured part: use Kronecker-like patterns
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if idx < n_structured:
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
                else:
                    break
    
    # Random part
    for r in range(n_structured, rank):
        U[:, r] = rng.normal(0, 1, n*m)
        V[:, r] = rng.normal(0, 1, m*p)
        W[:, r] = rng.normal(0, 1, p*n)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
    rng = np.random.RandomState(42)
    
    # Known: schoolbook rank is n*m*p = 24
    # Literature suggests rank 19 for 2x3 * 3x4
    # Try ranks from 19 upward
    
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    for rank in range(19, 25):
        if time.time() > deadline:
            break
        
        # Method 1: ALS with structured initialization
        U, V, W, err = _try_rank_als(n, m, p, rank, deadline, rng, max_restarts=20)
        if U is not None and err < 1e-8:
            # Check if half-integer
            if _check_half_integer(U, V, W, tol=1e-4):
                best_result = (U, V, W, rank)
                break
            else:
                # Try to round and verify
                Ur = _round_to_half_int(U)
                Vr = _round_to_half_int(V)
                Wr = _round_to_half_int(W)
                valid, max_err = _verify_fast(Ur, Vr, Wr, n, m, p, tol=1e-9)
                if valid:
                    best_result = (Ur, Vr, Wr, rank)
                    break
                else:
                    # Keep as floating point if it works
                    valid2, _ = _verify_fast(U, V, W, n, m, p, tol=1e-9)
                    if valid2:
                        best_result = (U, V, W, rank)
                        break
        
        # Method 2: ALS with random initialization (more restarts if time permits)
        if time.time() < deadline:
            U2, V2, W2, err2 = _try_rank_als(n, m, p, rank, deadline, rng, max_restarts=15)
            if U2 is not None and err2 < 1e-8:
                if _check_half_integer(U2, V2, W2, tol=1e-4):
                    best_result = (U2, V2, W2, rank)
                    break
                else:
                    Ur2 = _round_to_half_int(U2)
                    Vr2 = _round_to_half_int(V2)
                    Wr2 = _round_to_half_int(W2)
                    valid, _ = _verify_fast(Ur2, Vr2, Wr2, n, m, p, tol=1e-9)
                    if valid:
                        best_result = (Ur2, Vr2, Wr2, rank)
                        break
                    else:
                        valid2, _ = _verify_fast(U2, V2, W2, n, m, p, tol=1e-9)
                        if valid2:
                            best_result = (U2, V2, W2, rank)
                            break
        
        # Method 3: scipy optimization if time permits
        if time.time() < deadline - 30:
            U3, V3, W3, err3 = _try_rank_optimize(n, m, p, rank, deadline, rng, max_restarts=5)
            if U3 is not None and err3 < 1e-8:
                if _check_half_integer(U3, V3, W3, tol=1e-4):
                    best_result = (U3, V3, W3, rank)
                    break
                else:
                    Ur3 = _round_to_half_int(U3)
                    Vr3 = _round_to_half_int(V3)
                    Wr3 = _round_to_half_int(W3)
                    valid, _ = _verify_fast(Ur3, Vr3, Wr3, n, m, p, tol=1e-9)
                    if valid:
                        best_result = (Ur3, Vr3, Wr3, rank)
                        break
                    else:
                        valid2, _ = _verify_fast(U3, V3, W3, n, m, p, tol=1e-9)
                        if valid2:
                            best_result = (U3, V3, W3, rank)
                            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition
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

# EVOLVE-BLOCK-END
