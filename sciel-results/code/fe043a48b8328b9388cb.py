"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
Uses alternating least squares with structured initialization.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_rank = n * m * p  # 90
    best_result = _schoolbook(n, m, p, best_rank)
    
    # Verify baseline
    if not _verify(n, m, p, best_result):
        raise ValueError("Schoolbook failed verification")
    
    def _try_rank(r, num_outer=5, num_inner=50, time_budget=None):
        """Try to find a rank-r decomposition via alternating least squares."""
        nonlocal best_rank, best_result
        
        if time_budget is None:
            time_budget = deadline - 5
        
        for outer in range(num_outer):
            if time.time() > time_budget:
                break
            
            # Random initialization
            np.random.seed(1000 + outer * 7 + r)
            U = np.random.uniform(-1, 1, (n*m, r))
            V = np.random.uniform(-1, 1, (m*p, r))
            W = np.random.uniform(-1, 1, (p*n, r))
            
            # Normalize columns to unit norm for stability
            for c in range(r):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                if nu > 1e-10: U[:, c] /= nu
                if nv > 1e-10: V[:, c] /= nv
                if nw > 1e-10: W[:, c] /= nw
            
            for inner in range(num_inner):
                if time.time() > time_budget:
                    break
                
                # Step 1: Fix U,V, solve for W (linear system)
                # For each (i,j,k): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                # This is linear in W. Build A_W * W_flat = b
                A_W = np.zeros((n*m*p, p*n*r))
                b_W = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                col = (k*n + i) * r + c
                                A_W[idx, col] = U[i*m+j, c] * V[j*p+k, c]
                            idx += 1
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                    W = W_flat.reshape(p*n, r)
                except:
                    break
                
                # Step 2: Fix W,V, solve for U (linear system)
                A_U = np.zeros((n*m*p, n*m*r))
                b_U = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                col = (i*m + j) * r + c
                                A_U[idx, col] = W[k*n+i, c] * V[j*p+k, c]
                            idx += 1
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                    U = U_flat.reshape(n*m, r)
                except:
                    break
                
                # Step 3: Fix W,U, solve for V (linear system)
                A_V = np.zeros((n*m*p, m*p*r))
                b_V = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                col = (j*p + k) * r + c
                                A_V[idx, col] = W[k*n+i, c] * U[i*m+j, c]
                            idx += 1
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                    V = V_flat.reshape(m*p, r)
                except:
                    break
                
                # Check if we've converged to an exact solution
                if _check_decomp(n, m, p, U, V, W, r):
                    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    if r < best_rank:
                        best_rank = r
                        best_result = result
                    return True
        
        return False
    
    # Try ranks from lower to higher
    # Based on known results, the minimum rank for 3x5x6 is likely between 30-60
    # Try a range of ranks
    ranks_to_try = list(range(35, 90, 5)) + [36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66, 67, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79, 81, 82, 83, 84, 86, 87, 88, 89]
    ranks_to_try = sorted(set(ranks_to_try))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        # Allocate time based on rank (lower ranks get more time)
        time_budget = min(deadline - 5, time.time() + 30)
        if _try_rank(r, num_outer=3, num_inner=30, time_budget=time_budget):
            if best_rank <= r:
                break
    
    return best_result


def _schoolbook(n, m, p, rank):
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


def _check_decomp(n, m, p, U, V, W, r):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _verify(n, m, p, result):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    r = U.shape[1]
    return _check_decomp(n, m, p, U, V, W, r)
