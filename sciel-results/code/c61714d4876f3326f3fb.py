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

    def try_rank_als(R, max_iters=300, n_restarts=10, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart * 1000 + R)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            best_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row (i,j)
                # Vectorized update for U
                for i in range(n):
                    for j in range(m):
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # V[j] is (p, R), W[:, i] is (p, R)
                        M = V[j*p:(j+1)*p, :] * W[:, i*n:(i+1)*n].T # Wait, W is (p*n, R). W[k*n+i] is row k*n+i.
                        # We need M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # V_block = V[j*p:(j+1)*p, :] (p, R)
                        # W_block = W[:, :] is (p*n, R). We need rows k*n+i for k in 0..p-1.
                        # Indices: 0*n+i, 1*n+i, ... (p-1)*n+i
                        w_idx = [k*n + i for k in range(p)]
                        W_block = W[w_idx, :] # (p, R)
                        M = V[j*p:(j+1)*p, :] * W_block
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row (j,k)
                for j in range(m):
                    for k in range(p):
                        # N[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # U_block = U[:, j] (n, R) -> rows i*m+j
                        u_idx = [i*m + j for i in range(n)]
                        U_block = U[u_idx, :] # (n, R)
                        W_row = W[k*n:(k+1)*n, :] # (n, R)
                        N = U_block * W_row
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row (k,i)
                for k in range(p):
                    for i in range(n):
                        # P[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # U_row = U[i*m:(i+1)*m, :] (m, R)
                        U_row = U[i*m:(i+1)*m, :]
                        # V_col = V[:, k] (m, R) -> rows j*p+k
                        v_idx = [j*p + k for j in range(m)]
                        V_block = V[v_idx, :] # (m, R)
                        P = U_row * V_block
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

    # Strategy: Search likely ranks with more restarts
    # Based on 2x2x5 rank estimates, 2x4x5 rank is likely 28-36.
    # Try 20-25 with fewer restarts, 26-39 with more.
    
    search_plan = [
        (20, 2),
        (21, 2),
        (22, 2),
        (23, 3),
        (24, 3),
        (25, 3),
        (26, 5),
        (27, 5),
        (28, 8),
        (29, 8),
        (30, 10),
        (31, 10),
        (32, 12),
        (33, 12),
        (34, 12),
        (35, 10),
        (36, 8),
        (37, 5),
        (38, 5),
        (39, 5)
    ]
    
    for R, restarts in search_plan:
        if time.time() > deadline:
            break
        result = try_rank_als(R, max_iters=200, n_restarts=restarts, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
