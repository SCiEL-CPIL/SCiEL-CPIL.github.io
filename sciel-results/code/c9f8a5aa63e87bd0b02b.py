"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def try_rank(r, seed=42):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        if time.time() > deadline:
            return None
            
        rng = np.random.RandomState(seed)
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        
        # Strassen terms for 2x2x2 subproblem
        strassen_U = [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, -1, 0],
            [0, 1, 0, -1],
        ]
        strassen_V = [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, -1, 0],
            [1, -1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ]
        
        for c in range(min(7, r)):
            U[0, c] = strassen_U[c][0]
            U[1, c] = strassen_U[c][1]
            U[3, c] = strassen_U[c][2]
            U[4, c] = strassen_U[c][3]
            
            V[0, c] = strassen_V[c][0]
            V[1, c] = strassen_V[c][1]
            V[5, c] = strassen_V[c][2]
            V[6, c] = strassen_V[c][3]
        
        for c in range(7, r):
            U[:, c] = rng.choice([-1, 0, 1], size=n*m)
            V[:, c] = rng.choice([-1, 0, 1], size=m*p)
        
        num_eqs = n * m * p
        num_unk = p * n * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        unknown_idx = (k * n + i) * r + c
                        A[eq_idx, unknown_idx] = U[i*m+j, c] * V[j*p+k, c]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        
        W = W_flat.reshape(p*n, r)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-6:
            return None
        
        return U, V, W
    
    for r in range(20, 31):
        if time.time() > deadline:
            break
        for seed in range(100):
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
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
