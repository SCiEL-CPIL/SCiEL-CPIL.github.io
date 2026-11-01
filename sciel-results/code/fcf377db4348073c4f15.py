import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-39 decomposition for 2x2 * 2x11 matrix multiplication.
    
    Strategy: Split the 2x11 B matrix into blocks:
    - 4 blocks of 2 columns (cols 0-7): each uses Strassen rank 7
    - 1 block of 2 columns (cols 8-9): uses Strassen rank 7
    - 1 column (col 10): uses schoolbook rank 4
    Total: 5*7 + 4 = 39
    
    Actually, let's try a better split:
    - 3 blocks of 2 columns (cols 0-5): Strassen rank 7 each = 21
    - 2 blocks of 2 columns (cols 6-9): Strassen rank 7 each = 14  
    - 1 column (col 10): schoolbook rank 4
    Total: 21 + 14 + 4 = 39
    
    Or even better, can we do rank 38?
    Let's think: 2x2 * 2x11. The trivial rank is 4*11 = 44.
    Using Strassen on 2x2 blocks: each 2x2*2x2 costs 7, so 5 blocks of 2 cols = 35, plus 1 col of 2x2*2x1 = 4, total 39.
    
    Can we do better? What about overlapping blocks?
    
    Actually, for 2x2 * 2x11, the known minimal rank might be 38 or 39.
    Let me try to find a rank-38 decomposition.
    
    One approach: use the fact that we can share some Strassen terms.
    
    Let's try: 2x2 * 2x11 = (2x2 * 2x10) + (2x2 * 2x1)
    2x2 * 2x10: 5 blocks of 2x2*2x2, each rank 7 = 35
    2x2 * 2x1: rank 4
    Total: 39
    
    Alternative: Can we do 2x2 * 2x11 in rank 38?
    This would require saving 1 rank from the naive 39.
    
    Let me try a different decomposition. Consider the tensor as:
    sum over i,j,k of A[i,j] * B[j,k] * C[i,k]
    
    For rank r, we need U (4 x r), V (22 x r), W (22 x r) such that
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    Let me try to find a rank-38 solution using a search or known results.
    
    Actually, let me just implement the rank-39 solution cleanly and verify it works.
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indexing: A is 2x2, so A[i,j] with i in {0,1}, j in {0,1}
    # U index: i * m + j = i * 2 + j
    # B is 2x11, so B[j,k] with j in {0,1}, k in {0,...,10}
    # V index: j * p + k = j * 11 + k
    # C is 2x11, so C[i,k] with i in {0,1}, k in {0,...,10}
    # W index: k * n + i = k * 2 + i (per the trace convention W[k*n+i, r])
    
    def strassen_block(b, r0):
        """
        Strassen for A(2x2) * B[:, 2b:2b+2] -> C[:, 2b:2b+2]
        
        B columns: 2b, 2b+1
        C columns: 2b, 2b+1
        
        V indices: 
          B[0, 2b] = 0*11 + 2b = 2b
          B[1, 2b] = 1*11 + 2b = 11 + 2b
          B[0, 2b+1] = 2b + 1
          B[1, 2b+1] = 11 + 2b + 1 = 12 + 2b
        
        W indices (W[k*n+i, r] = W[2k+i, r]):
          C[0, 2b] = W[2*(2b) + 0] = W[4b]
          C[1, 2b] = W[2*(2b) + 1] = W[4b + 1]
          C[0, 2b+1] = W[2*(2b+1) + 0] = W[4b + 2]
          C[1, 2b+1] = W[2*(2b+1) + 1] = W[4b + 3]
        """
        v00 = 2 * b       # B[0, 2b]
        v01 = 2 * b + 1   # B[0, 2b+1]
        v10 = 11 + 2 * b  # B[1, 2b]
        v11 = 12 + 2 * b  # B[1, 2b+1]
        
        w00 = 4 * b       # C[0, 2b]
        w10 = 4 * b + 1   # C[1, 2b]
        w01 = 4 * b + 2   # C[0, 2b+1]
        w11 = 4 * b + 3   # C[1, 2b+1]
        
        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        
        r = r0
        # M1 = (A00 + A11)(B00 + B11)
        U[0, r] += 1; U[3, r] += 1
        V[v00, r] += 1; V[v11, r] += 1
        W[w00, r] += 1; W[w11, r] += 1
        r += 1
        
        # M2 = A11(B01 - B00)
        U[3, r] += 1
        V[v01, r] += 1; V[v00, r] -= 1
        W[w01, r] += 1; W[w11, r] += 1
        r += 1
        
        # M3 = A00(B10 - B11)
        U[0, r] += 1
        V[v10, r] += 1; V[v11, r] -= 1
        W[w00, r] += 1; W[w10, r] += 1
        r += 1
        
        # M4 = A11(B00 + B11)
        U[3, r] += 1
        V[v00, r] += 1; V[v11, r] += 1
        W[w11, r] += 1
        r += 1
        
        # M5 = (A00 + A01)B11
        U[0, r] += 1; U[1, r] += 1
        V[v11, r] += 1
        W[w01, r] += 1; W[w11, r] += 1
        r += 1
        
        # M6 = (A10 - A00)B00
        U[2, r] += 1; U[0, r] -= 1
        V[v00, r] += 1
        W[w10, r] += 1; W[w00, r] -= 1
        r += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, r] += 1; U[3, r] -= 1
        V[v10, r] += 1; V[v11, r] += 1
        W[w00, r] += 1; W[w10, r] += 1
        r += 1
        
        return r
    
    r = 0
    for b in range(5):
        r = strassen_block(b, r)
    
    # Col 10: schoolbook, 4 terms
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    
    # V indices for B[:,10]: B[0,10]=10, B[1,10]=21
    # W indices for C[:,10]: C[0,10]=W[20], C[1,10]=W[21]
    
    U[0, r] = 1; V[10, r] = 1; W[20, r] = 1
    r += 1
    U[1, r] = 1; V[21, r] = 1; W[20, r] = 1
    r += 1
    U[2, r] = 1; V[10, r] = 1; W[21, r] = 1
    r += 1
    U[3, r] = 1; V[21, r] = 1; W[21, r] = 1
    r += 1
    
    # Convert to lists of integers
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
