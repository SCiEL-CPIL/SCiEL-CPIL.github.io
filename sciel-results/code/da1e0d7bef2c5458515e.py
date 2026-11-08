"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def make_schoolbook():
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
    
    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_rank_als(R, max_iters=200, n_restarts=4, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart * 1000)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            best_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row (i,j)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row (j,k)
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row (k,i)
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        # Found near-exact solution, convert to fractions
                        U_frac = [[Fraction(U[i*m+j, r]).limit_denominator(1000000) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(V[j*p+k, r]).limit_denominator(1000000) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(W[k*n+i, r]).limit_denominator(1000000) for r in range(R)] for k in range(p) for i in range(n)]
                        
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
                
                if time.time() > deadline:
                    return None
    
    # Try ranks from 21 up to 39 (below schoolbook 40)
    for R in range(21, 40):
        if time.time() > deadline:
            break
        result = try_rank_als(R, max_iters=150, n_restarts=3, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
