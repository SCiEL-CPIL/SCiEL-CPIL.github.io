"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try ranks from a reasonable lower bound up to schoolbook
    for rank in range(20, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Time budget per rank
        time_per_rank = 8 if rank < 35 else 4
        rank_deadline = time.time() + time_per_rank
        
        result = _try_rank_als(n, m, p, rank, rank_deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank_als(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using alternating least squares."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n_m, rank) * 0.5
        V = np.random.randn(m_p, rank) * 0.5
        W = np.random.randn(p_n, rank) * 0.5
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        U[i * m + j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V[j * p + k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W[k * n + i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Check convergence
            if _verify(n, m, p, U, V, W):
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(n, m, p, U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Check the final solution after all iterations
        if _verify(n, m, p, U, V, W):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(n, m, p, U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
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
