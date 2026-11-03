"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(u).limit_denominator(10**6) for u in row] for row in U]
    Vf = [[Fraction(v).limit_denominator(10**6) for v in row] for row in V]
    Wf = [[Fraction(w).limit_denominator(10**6) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for c in range(rank):
                    actual += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _snap_to_half_integer(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return x

def _als_decompose(n, m, p, rank, max_iter=50, tol=1e-10, deadline=None, seed=0):
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W

def _try_rank(rank, deadline, n, m, p):
    for seed in range(20):
        if time.time() > deadline - 5:
            return None
        try:
            U, V, W = _als_decompose(n, m, p, rank, max_iter=20, deadline=deadline-5, seed=seed)
            U_snapped = [[_snap_to_half_integer(x) for x in row] for row in U]
            V_snapped = [[_snap_to_half_integer(x) for x in row] for row in V]
            W_snapped = [[_snap_to_half_integer(x) for x in row] for row in W]
            
            # Check half-integer
            ok = True
            for row in U_snapped:
                for x in row:
                    if abs(2*x - round(2*x)) > 1e-6:
                        ok = False; break
                if not ok: break
            if not ok: continue
            for row in V_snapped:
                for x in row:
                    if abs(2*x - round(2*x)) > 1e-6:
                        ok = False; break
                if not ok: break
            if not ok: continue
            for row in W_snapped:
                for x in row:
                    if abs(2*x - round(2*x)) > 1e-6:
                        ok = False; break
                if not ok: break
            
            if _verify(U_snapped, V_snapped, W_snapped):
                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
        except:
            continue
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    for rank in range(16, 49):
        if time.time() > deadline - 10:
            break
        result = _try_rank(rank, deadline, n, m, p)
        if result:
            return result
            
    return _schoolbook()
# EVOLVE-BLOCK-END
