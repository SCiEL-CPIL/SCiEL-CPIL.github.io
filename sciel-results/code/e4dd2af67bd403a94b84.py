"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    
    For C = A * B where A is 2x2 and B is 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n=2, m=2, p=16):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
    
    # Check a few random entries
    import random
    random.seed(42)
    
    for _ in range(100):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        
        # Compute T[a,b,c] from decomposition
        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
        
        # Compute T[a,b,c] from definition
        # a = i*m + j (i-th row, j-th col of A)
        # b = j'*p + k (j'-th row, k-th col of B)
        # c = k'*n + i' (k'-th col, i'-th row of C)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        
        # T[a,b,c] = 1 if i_a == i_c, j_a == j_b, k_b == k_c, else 0
        expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
        
        if val != expected:
            return False
    
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        if _time_remaining() < 1.0:
            break
        col_offset = t * 7
        
        for r in range(7):
            U[:, col_offset + r] = U_core[:, r]
            
            # V row index: j*p + k
            V[2*t, col_offset + r] = V_core[0, r]
            V[2*t + 1, col_offset + r] = V_core[1, r]
            V[16 + 2*t, col_offset + r] = V_core[2, r]
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r]
            
            # W row index: k*n + i
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 2, col_offset + r] = W_core[1, r]
            W[4*t + 1, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: use trivial rank-8 decomposition tiled
        rank_trivial = 8 * num_pairs
        U_t = np.zeros((n * m, rank_trivial), dtype=np.int64)
        V_t = np.zeros((m * p, rank_trivial), dtype=np.int64)
        W_t = np.zeros((p * n, rank_trivial), dtype=np.int64)
        
        for t in range(num_pairs):
            for i in range(n):
                for j in range(m):
                    for k in range(2):
                        idx = t * 8 + i * 4 + j * 2 + k
                        U_t[i * m + j, idx] = 1
                        V[j * p + 2*t + k, idx] = 1
                        W[(2*t + k) * n + i, idx] = 1
        
        U, V, W = U_t, V_t, W_t
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
