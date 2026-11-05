import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition with half-integer entries using Fractions"""
        U_frac = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in U]
        V_frac = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in V]
        W_frac = [[Fraction(int(round(val * 2))) / 2 for val in row] for row in W]
        
        # Check if all entries are half-integers
        for mat in [U_frac, V_frac, W_frac]:
            for row in mat:
                for val in row:
                    if val.denominator not in [1, 2]:
                        return False
        
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def tensor_error(U, V, W, r):
        """Compute tensor decomposition error"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    err += (s - 1.0)**2
        return err

    def try_rank(r, max_iters=300, seed=42):
        """Try to find rank-r decomposition using alternating least squares"""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Better initialization: use structured random with smaller scale
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U rows
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V rows
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W rows
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0:
                err = tensor_error(U, V, W, r)
                if err < 1e-8:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), r):
                        return (U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
        
        return None

    # Try ranks from 70 down to 55
    for r in range(70, 54, -1):
        if time.time() > deadline - 10:
            break
        for seed in [42, 123, 999, 777, 555]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=200, seed=seed)
            if result is not None:
                U, V, W, rank = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        if best_rank < 70:
            break
    
    return best_result
