"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
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
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=2000, tol=1e-8, seed=0):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest half-integer if close
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def _extend_decomposition(U, V, W, n, m, p_old, p_new):
    """
    Extend a decomposition of M_{n,m,p_old} to M_{n,m,p_new} where p_new is a multiple of p_old.
    """
    q = p_new // p_old
    if q * p_old != p_new:
        return None
    
    U_new = U
    V_new = [[0] * len(V[0]) for _ in range(m * p_new)]
    W_new = [[0] * len(W[0]) for _ in range(p_new * n)]
    
    for j in range(m):
        for k in range(p_new):
            k_old = k % p_old
            V_new[j * p_new + k] = V[j * p_old + k_old]
            
    for k in range(p_new):
        k_old = k % p_old
        for i in range(n):
            W_new[k * n + i] = W[k_old * n + i]
            
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p
    
    # Strategy 1: Find decomposition for 3x4x3 and extend to 3x4x6
    # Rank of 3x4x3 is at most 36.
    # If we find rank R for 3x4x3, we get rank R for 3x4x6.
    n_sub, m_sub, p_sub = 3, 4, 3
    # Try ranks 24 to 36
    for rank in range(24, 37):
        if time.time() > _DEADLINE - 5.0:
            break
        for seed in range(10):
            if time.time() > _DEADLINE - 5.0:
                break
            result = _try_rank(n_sub, m_sub, p_sub, rank, max_iter=500, seed=seed)
            if result is not None:
                U_sub, V_sub, W_sub = result
                if _verify(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub):
                    # Extend to 3x4x6
                    U_ext, V_ext, W_ext = _extend_decomposition(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, p)
                    if U_ext is not None and _verify(U_ext, V_ext, W_ext, n, m, p):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U_ext, "V": V_ext, "W": W_ext}
                        break
        if best_result is not None and best_rank < 36:
            break

    # Strategy 2: Direct search for 3x4x6 with ranks close to schoolbook
    if time.time() < _DEADLINE - 10.0:
        for rank in range(48, 72):
            if time.time() > _DEADLINE - 5.0:
                break
            for seed in range(2):
                if time.time() > _DEADLINE - 5.0:
                    break
                result = _try_rank(n, m, p, rank, max_iter=500, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
                        break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
