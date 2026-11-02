import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strategy: Use a rank-48 decomposition by combining Strassen on 2x2 blocks
    # with a clever sharing of factors across the 8 column-pairs of B.
    #
    # We decompose C = A*B where A is 2x4, B is 4x8.
    # Split A into A1 (cols 0-1) and A2 (cols 2-3), both 2x2.
    # Split B into B1 (rows 0-1) and B2 (rows 2-3), both 2x8.
    # C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x8 product, we use 4 Strassen 2x2*2x2 blocks.
    # But we can share the U and W factors across the two 2x2*2x8 products
    # by using a combined 4-term Strassen-like scheme.
    #
    # Actually, let's use a direct rank-48 construction:
    # For the 2x4 * 4x8 product, we can use 6 Strassen-like terms per 
    # "super-block" of 4x4 in the middle, but that doesn't quite work.
    #
    # Let me use a simpler approach: rank 56 with optimized code,
    # but try to find if rank 48 is achievable.
    
    # After analysis, I'll implement rank 56 with clean code and verify correctness.
    # The key is to ensure the decomposition is exact.
    
    # Strassen components for 2x2 matrix multiplication
    # U_local[i, l]: coefficient for A[i,j] in the l-th Strassen product
    # V_local[j, l]: coefficient for B[j,k] in the l-th Strassen product  
    # W_local[c, l]: coefficient for C[c] in the l-th Strassen product
    # where c = row*2 + col for the 2x2 output
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indexing: A[0]=A11, A[1]=A12, A[2]=A21, A[3]=A22
    #           B[0]=B11, B[1]=B12, B[2]=B21, B[3]=B22
    #           C[0]=C11, C[1]=C12, C[2]=C21, C[3]=C22
    
    # M1: (A11+A22)(B11+B22)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    # M2: (A21+A22)B11
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    
    # M3: A11(B21-B22)
    U_local[0, 2] = 1
    V_local[2, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    # M4: A22(B12-B11)
    U_local[3, 3] = 1
    V_local[1, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    # M5: (A11+A12)B22
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each of the 8 Strassen 2x2*2x2 sub-problems:
    # j = 0,1,2,3 (column-pair index in B)
    # s = 0,1 (which 2x2 block of A: s=0 uses cols 0-1, s=1 uses cols 2-3)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # A mapping: local index (0-3) -> global row in A (flattened 2x4)
            # A is 2x4, flattened as [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
            # Local 2x2 for s=0: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
            # Local 2x2 for s=1: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
            if s == 0:
                a_map = [0, 1, 4, 5]  # A11, A12, A21, A22
            else:
                a_map = [2, 3, 6, 7]
            
            # B mapping: local index (0-3) -> global position in B (flattened 4x8)
            # B is 4x8, flattened as [B[0,0]..B[0,7], B[1,0]..B[1,7], B[2,0]..B[2,7], B[3,0]..B[3,7]]
            # For s=0 (rows 0-1 of B): B11=B[0,2j], B12=B[0,2j+1], B21=B[1,2j], B22=B[1,2j+1]
            # For s=1 (rows 2-3 of B): B11=B[2,2j], B12=B[2,2j+1], B21=B[3,2j], B22=B[3,2j+1]
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
            
            # C mapping: local index (0-3) -> global position in C (flattened 2x8)
            # C is 2x8, W index is k*n+i = k*2+i (trace convention)
            # Local C11=C[0,2j], C12=C[0,2j+1], C21=C[1,2j], C22=C[1,2j+1]
            # W index: C[0,2j] -> (2j)*2+0 = 4j, C[0,2j+1] -> (2j+1)*2+0 = 4j+2
            #           C[1,2j] -> (2j)*2+1 = 4j+1, C[1,2j+1] -> (2j+1)*2+1 = 4j+3
            c_map = [4*j, 4*j+2, 4*j+1, 4*j+3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
