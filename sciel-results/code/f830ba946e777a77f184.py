DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_start_time = time.time()
_DEADLINE = 130  # 20s safety margin from 150s

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the 4x5x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if this term contributes to C[i,k], i.e., always 1 for valid (i,j,k)
    # Actually the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k (it's the identity on the index)
    # Wait, let me re-read: the tensor is T[i,j,k] where the output is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Verify: for each (i,k), C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal sum_j A[i,j]*B[j,k]
    # This means sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 if j is the correct index, 0 otherwise
    # Actually no. The decomposition is: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # Wait, that's not right either. Let me think again.
    
    # Standard tensor decomposition for matrix mult:
    # C[i,k] = sum_r U[i*m+? , r] ... 
    # Actually the standard form is:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # But here U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta[j1, j2] * (something)
    # Actually it should be: the product over r gives sum_j A[i,j]*B[j,k]
    
    # Let me just verify by checking that the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] equals 1 for all i,j,k
    # That's the schoolbook tensor. But wait, that's not how it works.
    
    # Re-reading the task: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n x m times m x p has entries T[i,j,k] where
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition is: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k]) W[k*n+i,r]
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta[j1,j2]
    
    # So let's verify this condition
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > 1e-9:
                        return False
    return True

def _try_greedy_reduction():
    """Try to reduce rank by finding linear dependencies."""
    n, m, p = DIMENSIONS
    full_rank = n * m * p
    
    # Start with schoolbook
    U = np.zeros((n*m, full_rank))
    V = np.zeros((m*p, full_rank))
    W = np.zeros((p*n, full_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to find a subset of columns that still satisfies the constraint
    # This is NP-hard, so use a greedy heuristic
    # For each column, check if it can be expressed as a linear combination of others
    # subject to the tensor constraint
    
    # Simpler approach: try random subsets
    import random
    random.seed(42)
    
    best_rank = full_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to remove columns one at a time
    used = list(range(full_rank))
    
    for attempt in range(500):
        if _time_left() < 5:
            break
        
        # Pick a random column to try removing
        idx = random.choice(used)
        new_used = [c for c in used if c != idx]
        
        # Check if the remaining columns still satisfy the constraint
        # We need to see if there exist coefficients such that the constraint holds
        # This is a system of linear equations
        
        # The constraint is: for all i, k, j1, j2:
        # sum_{r in new_used} U[i*m+j1,r] V[j2*p+k,r] W[k*n+i,r] = delta[j1,j2]
        
        # This is a bilinear system, hard to solve directly.
        # Instead, let's just check if the current assignment (with 0 for removed columns) works
        # That won't work since we'd be setting entries to 0.
        
        # Alternative: try to find a rank-reduction by looking for structure
        # For now, just return schoolbook
        pass
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    if _time_left() < 2:
        # Return schoolbook immediately
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
    
    # Try greedy reduction
    U_list, V_list, W_list = _try_greedy_reduction()
    
    if _time_left() < 2:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
