"""Program for matrix multiplication 2x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_half_integer(U, V, W, n, m, p):
    """Verify exactness after rounding to half-integers."""
    U2 = np.round(np.array(U) * 2) / 2.0
    V2 = np.round(np.array(V) * 2) / 2.0
    W2 = np.round(np.array(W) * 2) / 2.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U2[i * m + j] * V2[j * p + k], W2[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _als_decompose(n, m, p, r, seed=0, iters=2000, tol=1e-12):
    """ALS to find U,V,W of rank r such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]=1."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    err = float('inf')
    for it in range(iters):
        if _remaining() < 3.0:
            break
        
        # Solve for U: for each (i,j), find U[i*m+j,:] such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                b = np.ones(p)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = sol
        
        # Solve for V: for each (j,k), find V[j*p+k,:] such that
        # sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = sol
        
        # Solve for W: for each (k,i), find W[k*n+i,:] such that
        # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = sol
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                    err += (s - 1.0) ** 2
        
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(n, m, p, r, n_seeds=5):
    """Try to find a valid rank-r decomposition."""
    best_err = float('inf')
    best_UVW = None
    for seed in range(n_seeds):
        if _remaining() < 2.0:
            break
        U, V, W, err = _als_decompose(n, m, p, r, seed=seed, iters=1500)
        if err < best_err:
            best_err = err
            best_UVW = (U, V, W)
        if best_err < 1e-12:
            # Try rounding to half-integers
            if _verify_half_integer(U, V, W, n, m, p):
                U2 = np.round(U * 2) / 2.0
                V2 = np.round(V * 2) / 2.0
                W2 = np.round(W * 2) / 2.0
                return U2, V2, W2, True
            # Also check if it's already close enough without rounding
            if _verify(U, V, W, n, m, p):
                return U, V, W, True
    if best_UVW is not None and best_err < 1e-10:
        U, V, W = best_UVW
        if _verify(U, V, W, n, m, p):
            return U, V, W, True
    return None, None, None, False

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition, rank = n*m*p = 36."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def construct():
    """
    Return a valid tensor decomposition for 2x3x6 matrix multiplication.
    Searches for the minimum rank using ALS, falls back to schoolbook (rank 36).
    """
    n, m, p = DIMENSIONS
    
    # Theoretical lower bound: need at least max(n*p, m*p, n*m) = max(12,18,6) = 18? 
    # Actually the tensor has n*m*p = 36 nonzero entries but the constraint is weaker.
    # The rank must be at least the max dimension of any mode unfolding.
    # Mode-1 (i): n=2, Mode-2 (j): m=3, Mode-3 (k): p=6
    # But our tensor is not a standard 3-tensor; it's indexed as (i,j) x (j,k) x (k,i).
    # The "rank" in this bilinear sense: we need rank >= max(n, m, p) at minimum? No.
    # Actually, the bilinear complexity is at least max(n*p, m*p, n*m) for general matrices,
    # but for this specific structure it could be lower.
    # Let's try ranks from 6 upward.
    
    # Try increasing ranks
    for r in range(6, 37):
        if _remaining() < 5.0:
            break
        U, V, W, success = _try_rank(n, m, p, r, n_seeds=3)
        if success:
            # Convert to lists of lists with integer/half-integer entries
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
