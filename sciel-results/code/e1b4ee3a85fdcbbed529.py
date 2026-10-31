"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4 matrix multiplication
    # Using tensor product of two Strassen 2x2 decompositions
    # Rank = 7 * 7 = 49
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices, the 7 rank-1 terms have these coefficient patterns
    # We need to map this to the 4x4 case using block structure
    
    # Block indices: for 4x4, we have 2x2 blocks of size 2x2
    # Block (bi, bj) where bi, bj in {0, 1}
    # Element (i, j) in 4x4 belongs to block (i//2, j//2)
    # Within block, local index (i%2, j%2)
    
    # Strassen 2x2 coefficient tables
    # U2[i,j, s] for s in 0..6 (7 terms)
    # V2[j,k, s] for s in 0..6
    # W2[k,i, s] for s in 0..6
    
    # Strassen 2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U2 coefficients (for 2x2 A matrix, rows i=0,1 cols j=0,1):
    # Term 0 (M1): A00 + A11 -> U2[0,0]=1, U2[1,1]=1
    # Term 1 (M2): A10 + A11 -> U2[1,0]=1, U2[1,1]=1
    # Term 2 (M3): A00 -> U2[0,0]=1
    # Term 3 (M4): A11 -> U2[1,1]=1
    # Term 4 (M5): A00 + A01 -> U2[0,0]=1, U2[0,1]=1
    # Term 5 (M6): A10 - A00 -> U2[1,0]=1, U2[0,0]=-1
    # Term 6 (M7): A01 - A11 -> U2[0,1]=1, U2[1,1]=-1
    
    U2 = {}
    # Term 0
    U2[(0,0,0)] = 1
    U2[(1,1,0)] = 1
    # Term 1
    U2[(1,0,1)] = 1
    U2[(1,1,1)] = 1
    # Term 2
    U2[(0,0,2)] = 1
    # Term 3
    U2[(1,1,3)] = 1
    # Term 4
    U2[(0,0,4)] = 1
    U2[(0,1,4)] = 1
    # Term 5
    U2[(1,0,5)] = 1
    U2[(0,0,5)] = -1
    # Term 6
    U2[(0,1,6)] = 1
    U2[(1,1,6)] = -1
    
    # V2 coefficients (for 2x2 B matrix, rows j=0,1 cols k=0,1):
    # Term 0 (M1): B00 + B11 -> V2[0,0]=1, V2[1,1]=1
    # Term 1 (M2): B00 -> V2[0,0]=1
    # Term 2 (M3): B10 - B11 -> V2[1,0]=1, V2[1,1]=-1
    # Term 3 (M4): B01 - B00 -> V2[0,1]=1, V2[0,0]=-1
    # Term 4 (M5): B11 -> V2[1,1]=1
    # Term 5 (M6): B00 + B01 -> V2[0,0]=1, V2[0,1]=1
    # Term 6 (M7): B10 + B11 -> V2[1,0]=1, V2[1,1]=1
    
    V2 = {}
    # Term 0
    V2[(0,0,0)] = 1
    V2[(1,1,0)] = 1
    # Term 1
    V2[(0,0,1)] = 1
    # Term 2
    V2[(1,0,2)] = 1
    V2[(1,1,2)] = -1
    # Term 3
    V2[(0,1,3)] = 1
    V2[(0,0,3)] = -1
    # Term 4
    V2[(1,1,4)] = 1
    # Term 5
    V2[(0,0,5)] = 1
    V2[(0,1,5)] = 1
    # Term 6
    V2[(1,0,6)] = 1
    V2[(1,1,6)] = 1
    
    # W2 coefficients (for 2x2 C matrix, rows i=0,1 cols k=0,1):
    # C00 = M1 + M4 - M5 + M7 -> W2[0,0,0]=1, W2[0,0,3]=1, W2[0,0,4]=-1, W2[0,0,6]=1
    # C01 = M3 + M5 -> W2[0,1,2]=1, W2[0,1,4]=1
    # C10 = M2 + M4 -> W2[1,0,1]=1, W2[1,0,3]=1
    # C11 = M1 - M2 + M3 + M6 -> W2[1,1,0]=1, W2[1,1,1]=-1, W2[1,1,2]=1, W2[1,1,5]=1
    
    W2 = {}
    # C00
    W2[(0,0,0)] = 1
    W2[(0,0,3)] = 1
    W2[(0,0,4)] = -1
    W2[(0,0,6)] = 1
    # C01
    W2[(0,1,2)] = 1
    W2[(0,1,4)] = 1
    # C10
    W2[(1,0,1)] = 1
    W2[(1,0,3)] = 1
    # C11
    W2[(1,1,0)] = 1
    W2[(1,1,1)] = -1
    W2[(1,1,2)] = 1
    W2[(1,1,5)] = 1
    
    # Now build the 4x4 decomposition using tensor product structure
    # For 4x4, we have block indices (bi, bj) where bi, bj in {0,1}
    # Local indices (li, lj) where li, lj in {0,1}
    # Global index: i = 2*bi + li, j = 2*bj + lj
    
    # The 49 terms are indexed by (s, t) where s, t in 0..6
    # s is the Strassen term for the outer (block-level) multiplication
    # t is the Strassen term for the inner (2x2) multiplication
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # For each element position (i, j) in A (4x4):
            # i = 2*bi + li, j = 2*bj + lj
            for bi in range(2):
                for bj in range(2):
                    for li in range(2):
                        for lj in range(2):
                            i = 2 * bi + li
                            j = 2 * bj + lj
                            U_idx = i * 4 + j
                            
                            # Outer Strassen: block (bi, bj) with term s
                            # Inner Strassen: local (li, lj) with term t
                            u_outer = U2.get((bi, bj, s), 0)
                            u_inner = U2.get((li, lj, t), 0)
                            U[U_idx][col] += u_outer * u_inner
            
            # For each element position (j, k) in B (4x4):
            # j = 2*bj + lj, k = 2*bk + lk
            for bj in range(2):
                for bk in range(2):
                    for lj in range(2):
                        for lk in range(2):
                            j = 2 * bj + lj
                            k = 2 * bk + lk
                            V_idx = j * 4 + k
                            
                            # Outer Strassen: block (bj, bk) with term s
                            # Inner Strassen: local (lj, lk) with term t
                            v_outer = V2.get((bj, bk, s), 0)
                            v_inner = V2.get((lj, lk, t), 0)
                            V[V_idx][col] += v_outer * v_inner
            
            # For each element position (i, k) in C (4x4):
            # i = 2*bi + li, k = 2*bk + lk
            for bi in range(2):
                for bk in range(2):
                    for li in range(2):
                        for lk in range(2):
                            i = 2 * bi + li
                            k = 2 * bk + lk
                            W_idx = k * 4 + i  # W[k*n+i, r] where n=4
                            
                            # Outer Strassen: block (bi, bk) with term s
                            # Inner Strassen: local (li, lk) with term t
                            w_outer = W2.get((bi, bk, s), 0)
                            w_inner = W2.get((li, lk, t), 0)
                            W[W_idx][col] += w_outer * w_inner
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
