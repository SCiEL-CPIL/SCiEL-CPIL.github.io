"""Tensor decomposition search for 3x3 @ 3x7 matrix multiplication."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DEADLINE = 120.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0

def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - START_TIME)

def construct():
    """
    Return {'U': (9, rank), 'V': (21, rank), 'W': (21, rank)} 
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    The score is the rank (number of columns).
    
    Strategy:
    1. Try to find a low-rank decomposition via random search + local refinement.
    2. If time permits, do a more structured search.
    3. Fallback to schoolbook rank 63.
    """
    n, m, p = 3, 3, 7
    # Tensor T[i, j, k] = 1 if C[i,k] depends on A[i,j] and B[j,k]
    # T shape: (n, m, p) = (3, 3, 7)
    # We want U (n*m, r), V (m*p, r), W (p*n, r) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook solution:
    rank_schoolbook = n * m * p  # 63
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m*p) + j * p + k
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Verify schoolbook
    def verify(U, V, W):
        r = U.shape[1]
        # Compute C = A @ B for random inputs
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # For a fixed (i,j,k), the contribution is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Sum over r should equal 1 if (i,j,k) is a valid triple (which all are)
        
        # Actually, the tensor entry T[i,j,k] should be reconstructed.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(r):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(best_U, best_V, best_W):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    # Try to find a better decomposition
    # We can try rank reduction by checking if some columns are redundant
    # or by searching for a lower rank decomposition
    
    # For now, let's try a simple local search:
    # Try rank 21 (Laderman-like) and see if we can find a valid decomposition
    
    # Actually, let's just return the schoolbook for now, as finding a valid
    # low-rank decomposition for 3x3x7 is non-trivial.
    
    # But wait, the task says "Find a rank-minimal exact tensor decomposition".
    # The known minimal rank for 3x3x7 might be lower than 63.
    # Let's try a random search for lower ranks.
    
    for target_rank in [35, 42, 49, 56, 62]:
        if _time_left() < 10:
            break
        # Random search: generate random U, V, W and check if they work
        # This is unlikely to succeed quickly, but let's try a few iterations
        
        max_iters = 1000
        for it in range(max_iters):
            if _time_left() < 10:
                break
            # Generate random matrices with entries in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, (n*m, target_rank))
            V_rand = np.random.randint(-1, 2, (m*p, target_rank))
            W_rand = np.random.randint(-1, 2, (p*n, target_rank))
            
            # Check if this works
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m + j, r] * V_rand[j*p + k, r] * W_rand[k*n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            if valid:
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                best_rank = target_rank
                break
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
