import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # We'll try to find a rank-27 decomposition by combining:
    # - Rank-10 decomposition for 2x2x3 (j=0,1; k=0,1,2)
    # - Rank-7 Strassen for 2x2x2 (j=0,1; k=3,4)  
    # - Rank-6 schoolbook for j=2 contributions
    # But we need to be careful about the W index convention.
    
    # Let's use a direct constructive approach with rank 27.
    # 
    # The tensor T[i,j,k] = 1 for all i,j,k in valid ranges.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Strategy: 
    # Part A: j=0,1; k=0,1,2 -> 2x2x3 subproblem, rank 10
    # Part B: j=0,1; k=3,4 -> 2x2x2 subproblem, rank 7 (Strassen)
    # Part C: j=2; k=0,1,2,3,4 -> 2x1x5 subproblem, rank 10
    # Total: 10 + 7 + 10 = 27
    
    rank = 27
    U = np.zeros((n * m, rank))  # 6 x 27
    V = np.zeros((m * p, rank))  # 15 x 27
    W = np.zeros((p * n, rank))  # 10 x 27
    
    col = 0
    
    # Part A: 2x2x3 decomposition (rank 10)
    # For i in {0,1}, j in {0,1}, k in {0,1,2}
    # Known rank-10 terms for M_{2,2,3}:
    # We use the standard decomposition:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, col] = 1; V[0, col] = 1; W[0, col] = 1; col += 1
    # Term 1: A[1,0]*B[0,0] -> C[1,0]  
    U[1, col] = 1; V[0, col] = 1; W[1, col] = 1; col += 1
    # Term 2: A[0,0]*B[0,1] -> C[0,1]
    U[0, col] = 1; V[1, col] = 1; W[2, col] = 1; col += 1
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    U[1, col] = 1; V[1, col] = 1; W[3, col] = 1; col += 1
    # Term 4: A[0,0]*B[0,2] -> C[0,2]
    U[0, col] = 1; V[2, col] = 1; W[4, col] = 1; col += 1
    # Term 5: A[1,0]*B[0,2] -> C[1,2]
    U[1, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1
    # Term 6: A[0,1]*B[1,0] -> C[0,0]
    U[2, col] = 1; V[3, col] = 1; W[0, col] = 1; col += 1
    # Term 7: A[1,1]*B[1,0] -> C[1,0]
    U[3, col] = 1; V[3, col] = 1; W[1, col] = 1; col += 1
    # Term 8: A[0,1]*B[1,1] -> C[0,1]
    U[2, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    # Term 9: A[1,1]*B[1,1] -> C[1,1]
    U[3, col] = 1; V[4, col] = 1; W[3, col] = 1; col += 1
    # We're missing C[0,2] and C[1,2] from A[:,1]*B[1,2]
    # Term 10 would be needed but we only have 10 terms for rank 10...
    # This schoolbook approach gives rank 12 for 2x2x3, not 10.
    
    # Let me use the actual known rank-10 decomposition for 2x2x3.
    # The rank-10 decomposition uses Strassen-like tricks.
    
    # Reset and try a different approach: just use rank 28 (known good) 
    # but verify it works, since the current program already achieves this.
    
    # Actually, let me just replicate the known-good rank-28 Strassen approach
    # but with cleaner code and verification.
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        """Strassen for 2x2x2 block: A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1]"""
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = A22(B11-B21)
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = -1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = (A11-A21)B22
        U[i1*m+j1, col] = 1; U[i2*m+j1, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i1, col] = 1
        col += 1
        # M4 = A11(B22-B12)
        U[i1*m+j1, col] = 1
        V[j2*p+k2, col] = 1; V[j1*p+k2, col] = -1
        W[k1*n+i1, col] = 1; W[k2*n+i1, col] = -1
        col += 1
        # M5 = (A11+A12)B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = 1
        col += 1
    
    def schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Apply Strassen to (i=0,1) x (j=0,1) x (k=0,1)
    strassen_2x2x2(0, 1, 0, 1, 0, 1)
    # Apply Strassen to (i=0,1) x (j=0,1) x (k=2,3)
    strassen_2x2x2(0, 1, 0, 1, 2, 3)
    # Schoolbook for remaining: j=2 with all k, and j=0,1 with k=4
    schoolbook([0, 1], [2], [0, 1, 2, 3, 4])  # 10 terms
    schoolbook([0, 1], [0, 1], [4])  # 4 terms
    
    # Total: 7 + 7 + 10 + 4 = 28
    assert col == 28, f"Expected 28, got {col}"
    
    # Verify
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
