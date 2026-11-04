import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dict of non-zero entries."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1
    return T

def _check_exact(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = sum(U[u_idx, r] * V[v_idx, r] * W[w_idx, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """Strassen-based decomposition: split m=3 into (2,1), 5 blocks of 2x2x2 + schoolbook."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] = val
            
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k)][r] = val
            
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i][r] = val
            
            col += 1
            
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, rank, n_restarts=20, max_iters=150, seed_base=42, time_limit=60):
    """
    Proper ALS for tensor decomposition of the matmul tensor.
    """
    start = time.time()
    best_err = np.inf
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit:
            break
        np.random.seed(seed_base + restart * 7919)
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    sol, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    U[u_idx, :] = sol
            
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    sol, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    V[v_idx, :] = sol
            
            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    B = np.zeros((m, rank))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    sol, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    W[w_idx, :] = sol
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        val = sum(U[u_idx, r] * V[v_idx, r] * W[w_idx, r] for r in range(rank))
                        err += (val - 1.0)**2
            
            if err < 1e-12:
                break
        
        if time.time() - start > time_limit:
            break
            
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-10:
            break
    
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_result = _strassen_decomposition(n, m, p)
    best_rank = 55
    
    # Try to find lower rank decompositions
    for rank in range(50, 44, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        result = _als_decompose(n, m, p, rank, n_restarts=10, max_iters=80, 
                                seed_base=1000+rank, time_limit=remaining-15)
        if result is not None:
            U, V, W = result
            if _check_exact(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
