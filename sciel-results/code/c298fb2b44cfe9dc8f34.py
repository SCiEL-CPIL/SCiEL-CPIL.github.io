"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    r = len(U[0])
    # Convert to fractions
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(n*m)]
    Vf = [[Fraction(V[j][c]) for c in range(r)] for j in range(m*p)]
    Wf = [[Fraction(W[k][c]) for c in range(r)] for k in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = Fraction(0)
                    for c in range(r):
                        s += Wf[k*n+i][c] * Uf[i*m+j1][c] * Vf[j2*p+k][c]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-6):
    """Quick float verification."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(r):
                        s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _als_optimize(U, V, W, n, m, p, r, max_iters=100, deadline=None):
    """ALS optimization to minimize residual of bilinear identity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m*m, r))
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W[k*n+i, :] = w
        
        # Update U: for each (i,j1)
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U[i*m+j1, :] = u
        
        # Update V: for each (j2,k)
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2*p+k, :] = v
    
    return U, V, W

def _round_to_rational(arr, max_denom=8):
    """Round array entries to rationals with small denominators."""
    result = []
    for i in range(len(arr)):
        row = []
        for c in range(len(arr[0])):
            val = arr[i][c]
            best = round(val)
            best_err = abs(val - best)
            for d in range(2, max_denom+1):
                num = round(val * d)
                cand = num / d
                err = abs(val - cand)
                if err < best_err:
                    best = cand
                    best_err = err
            row.append(best)
        result.append(row)
    return result

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    np.random.seed(42)
    
    # Try ranks below schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    lower_bound = max(n*m, m*p, n*p)
    
    # Try a range of ranks
    ranks_to_try = list(range(lower_bound, rank0))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        n_restarts = 5 if r > rank0 - 20 else 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # ALS optimization
            U, V, W = _als_optimize(U, V, W, n, m, p, r, max_iters=80, deadline=deadline-10)
            
            # Check if close to exact
            if _verify_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
                # Try rounding to rationals
                U_r = _round_to_rational(U.tolist(), max_denom=4)
                V_r = _round_to_rational(V.tolist(), max_denom=4)
                W_r = _round_to_rational(W.tolist(), max_denom=4)
                
                if _verify_exact(U_r, V_r, W_r, n, m, p):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_r, V_r, W_r
                    break  # Found valid for this r, try smaller
        
        # If we found a valid decomposition at this rank, continue to smaller ranks
        # But if we've been trying for a while without success, move on
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
