"""Matrix multiplication 3x3 x 3x4 tensor decomposition search.
Target: minimize rank R such that multiplication tensor T[i,j,k,c] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t].
Uses local search with random restarts to find a low-rank exact (or near-exact) decomposition.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
NM, MP, PN = N*M, M*P, P*N  # 9, 12, 12

# Deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 100.0  # 100s margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def _build_tensor():
    """Build the 3x3x4 multiplication tensor T[i,j,k] for (AB)[i,k] = sum_j A[i,j] B[j,k].
    Actually the tensor has indices (i, j, k) where i in [0,N), j in [0,M), k in [0,P).
    T[i,j,k] = 1 if this term appears in the product, else 0.
    For matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    So T[i,j,k] = 1 for all valid (i,j,k).
    """
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0
    return T

def _matvec_decompose(U, V, W):
    """Compute the tensor reconstructed from factor matrices.
    U: (N*M, R), V: (M*P, R), W: (P*N, R)
    T_recon[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    R = U.shape[1]
    T_recon = np.zeros((N, M, P), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for t in range(R):
                    s += U[i*M + j, t] * V[j*P + k, t] * W[k*N + i, t]
                T_recon[i, j, k] = s
    return T_recon

def _error(U, V, W, T):
    """Compute relative error between reconstructed tensor and target."""
    T_recon = _matvec_decompose(U, V, W)
    return np.linalg.norm(T_recon - T) / max(np.linalg.norm(T), 1e-12)

def _local_search(U, V, W, T, max_iter=20000, tol=1e-10, step_size=0.1):
    """Gradient-free local search (Nelder-Mead style) to minimize error.
    Returns best (U, V, W, error) found."""
    R = U.shape[1]
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_err = _error(best_U, best_V, best_W, T)
    
    if best_err < tol:
        return best_U, best_V, best_W, best_err
    
    # Simple coordinate descent with random restarts within local search
    np.random.seed(int(time.time() * 1000) % 2**31)
    
    current_err = best_err
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        # Pick a random matrix and a random element
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            row = np.random.randint(NM)
            col = np.random.randint(R)
            old_val = U_cur[row, col]
            U_cur[row, col] = old_val + (np.random.rand() - 0.5) * step_size
            new_err = _error(U_cur, V_cur, W_cur, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
            else:
                U_cur[row, col] = old_val
        elif mat_idx == 1:
            row = np.random.randint(MP)
            col = np.random.randint(R)
            old_val = V_cur[row, col]
            V_cur[row, col] = old_val + (np.random.rand() - 0.5) * step_size
            new_err = _error(U_cur, V_cur, W_cur, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
            else:
                V_cur[row, col] = old_val
        else:
            row = np.random.randint(PN)
            col = np.random.randint(R)
            old_val = W_cur[row, col]
            W_cur[row, col] = old_val + (np.random.rand() - 0.5) * step_size
            new_err = _error(U_cur, V_cur, W_cur, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
            else:
                W_cur[row, col] = old_val
        
        # Occasionally adjust step size
        if iteration % 1000 == 0:
            step_size *= 0.9 if current_err < best_err * 0.99 else 1.1
            step_size = max(0.001, min(0.5, step_size))
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank(R, T, n_restarts=5):
    """Try to find a rank-R decomposition with random initialization and local search."""
    np.random.seed(42 + R)
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if _time_remaining() < 10.0:
            break
        
        # Random initialization
        U = np.random.randn(NM, R) * 0.1
        V = np.random.randn(MP, R) * 0.1
        W = np.random.randn(PN, R) * 0.1
        
        # Scale to match magnitudes
        # Target tensor has all entries = 1, sum of products should give 1
        # With R terms each ~0.1^3 * R = R * 0.001, we need R * 0.001 ~ 1, so scale up
        scale = (R * 0.001)**(-1/6)  # rough heuristic
        U *= scale
        V *= scale
        W *= scale
        
        U, V, W, err = _local_search(U, V, W, T, max_iter=5000, tol=1e-10)
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W, err)
        
        if best_err < 1e-10:
            break
    
    return best_result

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
    T = _build_tensor()
    
    # Start from schoolbook rank and try to reduce
    schoolbook_rank = N * M * P  # 36
    
    # Try various ranks in increasing order
    # Known theoretical bounds: rank >= max(N*M, M*P, P*N, N+M+P-2) = max(9,12,12,6) = 12
    # Practical known: rank 23 for 3x3x4
    
    for R in range(12, schoolbook_rank + 1):
        if _time_remaining() < 30.0:
            break
        
        if _time_remaining() < 5.0:
            break
        
        result = _try_rank(R, T, n_restarts=3)
        
        if result is not None:
            U, V, W, err = result
            if err < 1e-8:
                # Verify and return
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j, col] = 1.0
                V[j*P + k, col] = 1.0
                W[k*N + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
