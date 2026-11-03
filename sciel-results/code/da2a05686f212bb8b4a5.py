import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 4]
_DEADLINE = time.time() + 130.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] * A[i,j] * B[j,k]
                C_computed[i,k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # First, establish the schoolbook as baseline
    best_result, best_rank = _schoolbook(n, m, p)
    
    if time.time() > _DEADLINE - 10.0:
        return best_result
    
    # Attempt to find a lower-rank decomposition
    # We'll try ranks from 32 up to 47 (below schoolbook's 48)
    
    target_ranks = [40, 42, 44, 46, 47] # Start higher to increase chance of success, then go lower
    
    for target_rank in target_ranks:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > _DEADLINE - 15.0:
                break
            
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            max_iters = 50
            for it in range(max_iters):
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Fix U and V, solve for W
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros(target_rank)
                        for j in range(m):
                            C_ik += U[i*m+j, :] * V[j*p+k, :]
                        norm_sq = np.dot(C_ik, C_ik)
                        if norm_sq > 1e-15:
                            W[k*n+i, :] = C_ik / norm_sq
                        else:
                            W[k*n+i, :] = 0.01 * np.ones(target_rank)
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        C_jk = np.zeros(target_rank)
                        for i in range(n):
                            C_jk += U[i*m+j, :] * W[k*n+i, :]
                        norm_sq = np.dot(C_jk, C_jk)
                        if norm_sq > 1e-15:
                            V[j*p+k, :] = C_jk / norm_sq
                        else:
                            V[j*p+k, :] = 0.01 * np.ones(target_rank)
                
                # Fix V and W, solve for U
                for i in range(n):
                    for j in range(m):
                        C_ij = np.zeros(target_rank)
                        for k in range(p):
                            C_ij += V[j*p+k, :] * W[k*n+i, :]
                        norm_sq = np.dot(C_ij, C_ij)
                        if norm_sq > 1e-15:
                            U[i*m+j, :] = C_ij / norm_sq
                        else:
                            U[i*m+j, :] = 0.01 * np.ones(target_rank)
                
                # Check convergence
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-10:
                    # Convert to lists of floats for JSON compatibility
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if _verify(U_list, V_list, W_list, n, m, p, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                if max_res < 1e-15:
                    break  # Converged but didn't pass verify? Unlikely, but break anyway
        
    # Fallback to schoolbook
    return best_result
