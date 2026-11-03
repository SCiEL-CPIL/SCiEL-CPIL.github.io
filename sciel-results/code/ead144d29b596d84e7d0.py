"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 2 by 3 matrix by a 3 by 6 matrix.

Uses Kronecker product structure: 2x3x6 = (2x3x1) ⊗ (1x1x6).
The 1x1x6 tensor has minimal rank 6. If 2x3x1 has rank r1, then
2x3x6 has rank at most r1*6. We search for the minimal rank.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication.
    
    C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] * A[i',j]) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
    must equal sum_j A[i,j] * B[j,k] for all A, B.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for c in range(rank):
            L = np.zeros((n,))
            R = np.zeros((p,))
            for i in range(n):
                for j in range(m):
                    L[i] += U_arr[i * m + j, c] * A[i, j]
            for j in range(m):
                for k in range(p):
                    R[k] += V_arr[j * p + k, c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W_arr[k * n + i, c] * L[i] * R[k]
        
        if not np.allclose(C_comp, C_true, atol=tol, rtol=tol):
            return False
    return True

def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank using random search + refinement."""
    if time.time() > deadline:
        return None
    
    # Use structured initialization based on Kronecker product
    # 2x3x6 = (2x3x1) ⊗ (1x1x6)
    # For 1x1x6: rank 6, U=[1,1,1,1,1,1], V=I_6, W=I_6
    # For 2x3x1: try various ranks
    
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # Try ranks from target_rank up to schoolbook
    for r in range(target_rank, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers/half-integers
            np.random.seed(attempt * 1000 + r)
            U = np.random.randint(-2, 3, size=(n * m, r)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, r)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, r)) * 0.5
            
            # Snap to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                # Check if entries are integer or half-integer
                valid = True
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for val in row:
                            if val * 2 != int(val * 2):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    # Based on Kronecker product structure, try rank 24 first (4*6)
    result = _try_rank(n, m, p, 24, deadline)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
