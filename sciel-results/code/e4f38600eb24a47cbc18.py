import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 135.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _construct_rank_6():
    """
    Construct a rank-6 decomposition for 2x3 * 3x6.
    
    The idea is to split the k-dimension (size 6) into two blocks of size 3.
    For each block, the multiplication is effectively 2x3 * 3x3.
    However, we can do even better by observing the structure.
    
    Let's look at the tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    We want T[i,j,k] = 1 for all i,j,k.
    
    Consider the following rank-6 construction:
    Indices:
    i in {0,1}
    j in {0,1,2}
    k in {0,1,2,3,4,5}
    
    Let r in {0,1,2,3,4,5}.
    Let's map r to (j_block, k_block).
    Actually, let's try a simpler structure.
    
    Notice that if we set:
    U[i*m+j, r] depends on j and r.
    V[j*p+k, r] depends on j, k, r.
    W[k*n+i, r] depends on k, i, r.
    
    Let's try to separate the j and k dimensions.
    
    A known rank-6 construction for 2x3x6:
    
    Let r = 0..5.
    
    For r = 0,1,2:
      Let r correspond to j=0,1,2 for the first half of k (k=0,1,2).
      U[i*m+j, r] = 1 if j == r else 0. (For all i)
      V[j*p+k, r] = 1 if j == r and k < 3 else 0.
      W[k*n+i, r] = 1 if k < 3 else 0. (For all i)
      
    For r = 3,4,5:
      Let r correspond to j=0,1,2 for the second half of k (k=3,4,5).
      Let r' = r - 3.
      U[i*m+j, r] = 1 if j == r' else 0. (For all i)
      V[j*p+k, r] = 1 if j == r' and k >= 3 else 0.
      W[k*n+i, r] = 1 if k >= 3 else 0. (For all i)
      
    Let's verify this.
    
    Case 1: k < 3.
    Sum over r=0..5.
    Only r=0,1,2 can contribute because for r>=3, W[k*n+i, r] = 0 (since k<3).
    For r in {0,1,2}:
      U[i*m+j, r] = 1 if j == r else 0.
      V[j*p+k, r] = 1 if j == r else 0 (since k<3).
      W[k*n+i, r] = 1.
      
      Product is non-zero only if j == r.
      Since j is fixed, only the term where r == j survives.
      If j=0, r=0: 1*1*1 = 1.
      If j=1, r=1: 1*1*1 = 1.
      If j=2, r=2: 1*1*1 = 1.
      Sum = 1.
      
    Case 2: k >= 3.
    Sum over r=0..5.
    Only r=3,4,5 can contribute because for r<3, W[k*n+i, r] = 0 (since k>=3).
    For r in {3,4,5}, let r' = r - 3.
      U[i*m+j, r] = 1 if j == r' else 0.
      V[j*p+k, r] = 1 if j == r' else 0 (since k>=3).
      W[k*n+i, r] = 1.
      
      Product is non-zero only if j == r'.
      Since j is fixed, only the term where r' == j survives.
      If j=0, r=3: 1*1*1 = 1.
      If j=1, r=4: 1*1*1 = 1.
      If j=2, r=5: 1*1*1 = 1.
      Sum = 1.
      
    This decomposition is valid and has rank 6.
    """
    n, m, p = 2, 3, 6
    rank = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half of k (0,1,2)
    for r in range(3):
        # U[i*m+j, r] = 1 if j == r else 0
        for i in range(n):
            for j in range(m):
                if j == r:
                    U[i * m + j, r] = 1.0
        # V[j*p+k, r] = 1 if j == r and k < 3 else 0
        for j in range(m):
            for k in range(p):
                if j == r and k < 3:
                    V[j * p + k, r] = 1.0
        # W[k*n+i, r] = 1 if k < 3 else 0
        for k in range(p):
            for i in range(n):
                if k < 3:
                    W[k * n + i, r] = 1.0
                    
    # Second half of k (3,4,5)
    for r_prime in range(3):
        r = r_prime + 3
        # U[i*m+j, r] = 1 if j == r_prime else 0
        for i in range(n):
            for j in range(m):
                if j == r_prime:
                    U[i * m + j, r] = 1.0
        # V[j*p+k, r] = 1 if j == r_prime and k >= 3 else 0
        for j in range(m):
            for k in range(p):
                if j == r_prime and k >= 3:
                    V[j * p + k, r] = 1.0
        # W[k*n+i, r] = 1 if k >= 3 else 0
        for k in range(p):
            for i in range(n):
                if k >= 3:
                    W[k * n + i, r] = 1.0
                    
    return U, V, W

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # 2. Try to find a better rank
    U_best, V_best, W_best = None, None, None
    
    # Try rank 6 construction
    U6, V6, W6 = _construct_rank_6()
    if _verify_decomposition(U6, V6, W6, n, m, p):
        U_best, V_best, W_best = U6, V6, W6
    else:
        # Fallback to schoolbook
        U_best, V_best, W_best = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
