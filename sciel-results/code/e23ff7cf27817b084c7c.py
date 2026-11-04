"""Matrix multiplication 3x4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rationalize(arr, max_den=10000):
    """Try to convert floating point array to exact fractions."""
    result = []
    for row in arr:
        rrow = []
        for val in row:
            if abs(val) < 1e-12:
                rrow.append(0)
            else:
                frac = Fraction(val).limit_denominator(max_den)
                if abs(float(frac) - val) < 1e-10:
                    rrow.append(frac)
                else:
                    rrow.append(val)
        result.append(rrow)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _als_solve(n, m, p, r, max_iters=3000, seed=42):
    """
    Alternating least squares for rank-r decomposition.
    T[i,j,k] = 1 for all (i,j,k).
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, r) * 0.3
    V = np.random.randn(m * p, r) * 0.3
    W = np.random.randn(p * n, r) * 0.3
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-10:
                break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 59 down to 40
    for r in range(59, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(30):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_solve(n, m, p, r, max_iters=2000, seed=seed)
            
            if err < 1e-8:
                # Verify
                U_l, V_l, W_l = U.tolist(), V.tolist(), W.tolist()
                if _verify(U_l, V_l, W_l, n, m, p, tol=1e-6):
                    # Rationalize
                    U_r = _rationalize(U_l, max_den=100000)
                    V_r = _rationalize(V_l, max_den=100000)
                    W_r = _rationalize(W_l, max_den=100000)
                    
                    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-10):
                        return {"U": U_r, "V": V_r, "W": W_r}
                    else:
                        return {"U": U_l, "V": V_l, "W": W_l}
        
    return best_result

# EVOLVE-BLOCK-END
