import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _structured_decomposition(n, m, p):
    """
    Structured decomposition for 2x5 x 5x6.
    
    Key insight: For n=2, we can exploit the structure more effectively.
    The tensor T[i,j,k] = 1 for all (i,j,k) needs to be decomposed.
    
    Strategy: Use the fact that for each fixed j, T[i,j,k] is separable in i and k.
    We can try to share terms across different j values.
    
    Approach: 
    - For each pair of columns (k, k') in B, define combinations that allow us to
      compute multiple C[i,k] entries with fewer multiplications.
    
    Let's try: group the 6 columns of B into 3 pairs.
    For pair (k, k+1):
      Define S = B[:,k] + B[:,k+1], D = B[:,k] - B[:,k+1]
      Then:
        A[i,:]·B[:,k] = (A[i,:]·S + A[i,:]·B[:,k]) / 2  -- not helpful directly
      
    Better approach: Use a rank-2 decomposition per "block"
    
    Actually, let me try a direct construction:
    
    For 2x5 x 5x6, consider:
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + ... + A[i,4]*B[4,k]
    
    That's 5 terms per (i,k), so 2*6*5 = 60 total (schoolbook).
    
    To reduce rank, we need to find linear dependencies.
    
    Let me try a different structured approach using the small n=2:
    
    Define for each j:
      u_j = (A[0,j], A[1,j])  (2-vector)
      v_j = (B[j,0], B[j,1], ..., B[j,5])  (6-vector)
    
    Then C[i,k] = sum_j u_j[i] * v_j[k]
    
    This is a sum of 5 rank-1 matrices (in the i-k space), each scaled by...
    wait, that's not quite right.
    
    Actually, C = A*B where A is 2x5 and B is 5x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can view this as: for each j, the outer product of column j of A (as a 2-vector)
    and row j of B (as a 6-vector) contributes to C.
    
    So C = sum_j (A[:,j] * B[j,:])  -- sum of 5 rank-1 matrices
    
    But we need a tensor decomposition, not a matrix decomposition.
    
    The tensor T[i,j,k] has the property that T[i,j,k] = 1 for all i,j,k.
    We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Let me try a greedy approach: start with schoolbook and try to merge terms.
    """
    # Start with schoolbook
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _greedy_low_rank(n, m, p, target_rank, max_iters=1000, seed=None):
    """
    Greedy search for a rank-r decomposition.
    
    Strategy: 
    1. Start with random sparse matrices U, V, W
    2. Use iterative refinement to satisfy the tensor equations
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Random initialization
        for r in range(target_rank):
            # Sparse random values
            for idx in range(n*m):
                if random.random() < 0.2:
                    U[idx, r] = random.choice([-1, 1])
            for idx in range(m*p):
                if random.random() < 0.2:
                    V[idx, r] = random.choice([-1, 1])
            for idx in range(p*n):
                if random.random() < 0.2:
                    W[idx, r] = random.choice([-1, 1])
        
        # Check if this works
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W, 0.0
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_error

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Start with schoolbook as baseline
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Try ranks from 30 upwards
    
    for target_rank in range(30, 60):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try greedy search
        U_try, V_try, W_try, err = _greedy_low_rank(n, m, p, target_rank, max_iters=500, seed=target_rank)
        
        if err < 1e-9:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
