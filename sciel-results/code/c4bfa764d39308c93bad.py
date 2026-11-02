"""Rewritten program for matrix multiplication 2x3 by 3x4 using random search with exact W solve."""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Try ranks from 12 upward
    for rank in range(12, 25):
        if time.time() > deadline:
            break
        result = _find_decomposition(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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


def _find_decomposition(rank, n, m, p, deadline):
    """
    Find a rank-r decomposition using random search with exact linear solve for W.
    """
    # Candidate values: integers and half-integers
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Number of random restarts, adjusted for time
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:  # 5s safety margin
            break
        
        # Random initialization of U and V
        U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n * m)])
        V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m * p)])
        
        # Solve for W: for each (k, i), we need
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
        W = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build the linear system: for each j in 0..m-1
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                b_sys = np.ones(m)
                
                # Solve A_sys * W[k*n+i, :]^T = b_sys
                try:
                    # Use lstsq for general case
                    W[k * n + i, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    
                    # Check if solution is valid (residuals should be ~0)
                    if len(residuals) > 0 and residuals[0] > 1e-10:
                        valid = False
                        break
                    
                    # Check if W entries are half-integers
                    for c in range(rank):
                        val = W[k * n + i, c]
                        # Check if val is close to a half-integer
                        nearest = round(val * 2) / 2
                        if abs(val - nearest) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the decomposition
        if _is_valid(U, V, W, n, m, p):
            # Round to exact half-integers
            W_exact = np.round(W * 2) / 2
            U_exact = np.round(U * 2) / 2
            V_exact = np.round(V * 2) / 2
            return _convert_to_list(U_exact, V_exact, W_exact)
    
    return None


def _is_valid(U, V, W, n, m, p):
    """
    Check if the decomposition is valid:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _convert_to_list(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
