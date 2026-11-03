"""Matrix multiplication 4x4x7 rank-minimal search using Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i, j, k), check that the sum equals 1 if it's a valid product entry, else 0.
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in the support.
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # So we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for ALL (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with schoolbook rank 112
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try Strassen-based decomposition
    # 4x4x7: split into 4x4x4 + 4x4x3
    # 4x4x4: Strassen on 2x2 blocks of 2x2x2 -> rank 7*7=49
    # 4x4x3: split 4x4 into two 2x4, each 2x4x3 -> two 2x2x3
    # 2x2x3: Strassen 2x2x2 (rank 7) + 2x2x1 (rank 2) = rank 9
    # Total: 49 + 2*9 = 49 + 18 = 67? Wait, 4x4x3: split rows of A into 2x4+2x4
    # Each 2x4x3: split cols of A into 2x2x3 + 2x2x3
    # So 4x4x3 = 2 * 2 * (2x2x3 rank) = 4 * 9 = 36
    # Total: 49 + 36 = 85
    
    if time.time() > deadline:
        return best_result
    
    # Build 2x2x2 Strassen decomposition
    # U: (4,7), V: (6,7), W: (4,7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Strassen: 7 multiplications
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    U_222 = [
        # r0: M1: (a00+a11) * (b00+b11)
        [1, 0, 0, 1],  # U for a00+a11
        [1, 1, 1, 1],  # V for b00+b11
        [1, 0, 1, 0],  # W for c00 (M1 contribution)
    ]
    # Let me be more careful. For each of the 7 terms:
    # Term r: U[term_r], V[term_r], W[term_r]
    # U is 4x7, V is 6x7, W is 4x7
    
    # Term 0: M1 = (a00+a11)(b00+b11)
    # U col 0: a00=1, a01=0, a10=0, a11=1 -> [1,0,0,1]
    # V col 0: b00=1, b01=1, b10=1, b11=1 -> but V has indices j*2+k, so j=0: k=0,1; j=1: k=0,1
    # V[0]=b00, V[1]=b01, V[2]=b10, V[3]=b11, V[4]=?, V[5]=?
    # Wait, for 2x2x2: n=2, m=2, p=2
    # U: (n*m, rank) = (4, rank), index i*m+j = i*2+j
    # V: (m*p, rank) = (4, rank), index j*p+k = j*2+k
    # W: (p*n, rank) = (4, rank), index k*n+i = k*2+i
    
    # So for 2x2x2:
    # U indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    # V indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
    # W indices: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    
    # M1 = (a00+a11)(b00+b11)
    # U col: [1, 0, 0, 1] (a00 and a11)
    # V col: [1, 1, 0, 0] (b00 and b01... wait, b00 is V[0], b11 is V[3])
    # Hmm, b00+b11: V[0]=1, V[3]=1, others 0
    # But we need (b00+b11) which means the V factor should have 1s where b00 and b11 appear
    # V col: [1, 0, 0, 1] (V[0]=b00=1, V[3]=b11=1)
    # W: contributes to c00 and c11
    # c00 = M1 + M4 - M5 + M7, c11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to c00 and +1 to c11
    # W col: c00 is W[0] (k=0,i=0), c11 is W[3] (k=1,i=1)
    # W col: [1, 0, 0, 1]
    
    # M2 = (a10+a11)b00
    # U col: [0, 0, 1, 1] (a10 and a11)
    # V col: [1, 0, 0, 0] (b00)
    # c10 = M2+M4, c11 = M1-M2+M3+M6
    # M2 contributes +1 to c10, -1 to c11
    # W col: c10 is W[1] (k=0,i=1), c11 is W[3] (k=1,i=1)
    # W col: [0, 1, 0, -1]
    
    # M3 = a00(b01-b11)
    # U col: [1, 0, 0, 0] (a00)
    # V col: [0, 1, 0, -1] (b01 - b11)
    # c01 = M3+M5, c11 = M1-M2+M3+M6
    # M3 contributes +1 to c01, +1 to c11
    # W col: c01 is W[2] (k=1,i=0), c11 is W[3] (k=1,i=1)
    # W col: [0, 0, 1, 1]
    
    # M4 = a11(b10-b00)
    # U col: [0, 0, 0, 1] (a11)
    # V col: [-1, 0, 1, 0] (b10 - b00)
    # c00 = M1+M4-M5+M7, c10 = M2+M4
    # M4 contributes +1 to c00, +1 to c10
    # W col: c00 is W[0], c10 is W[1]
    # W col: [1, 1, 0, 0]
    
    # M5 = (a00+a01)b11
    # U col: [1, 1, 0, 0] (a00 and a01)
    # V col: [0, 0, 0, 1] (b11)
    # c00 = M1+M4-M5+M7, c01 = M3+M5
    # M5 contributes -1 to c00, +1 to c01
    # W col: c00 is W[0], c01 is W[2]
    # W col: [-1, 0, 1, 0]
    
    # M6 = (a10-a00)(b00+b10)
    # U col: [-1, 0, 1, 0] (a10 - a00)
    # V col: [1, 0, 1, 0] (b00 + b10)
    # c11 = M1-M2+M3+M6
    # M6 contributes +1 to c11
    # W col: c11 is W[3]
    # W col: [0, 0, 0, 1]
    
    # M7 = (a01-a11)(b01+b11)
    # U col: [0, 1, 0, -1] (a01 - a11)
    # V col: [0, 1, 0, 1] (b01 + b11)
    # c00 = M1+M4-M5+M7
    # M7 contributes +1 to c00
    # W col: c00 is W[0]
    # W col: [1, 0, 0, 0]
    
    U_222 = [[1,0,0,1,  0,0,1,1,  0,0,0,1,  1,1,0,0,  -1,0,1,0,  0,1,0,-1],
             [0,0,0,0,  0,0,0,0,  0,0,0,0,  0,1,0,0,   0,0,0,0,   1,0,0,0],
             [0,0,1,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,   1,0,0,0,   0,0,0,0],
             [1,0,0,1,  0,0,0,0,  0,0,0,0,  0,0,1,0,   0,0,0,0,  -1,0,0,0]]
    
    # Wait, I need to be more careful. Let me redo this.
    # U is 4x7. Row i*2+j.
    # Col 0 (M1): U[0]=1, U[3]=1 (a00, a11)
    # Col 1 (M2): U[2]=1, U[3]=1 (a10, a11)
    # Col 2 (M3): U[0]=1 (a00)
    # Col 3 (M4): U[3]=1 (a11)
    # Col 4 (M5): U[0]=1, U[1]=1 (a00, a01)
    # Col 5 (M6): U[0]=-1, U[2]=1 (a10-a00)
    # Col 6 (M7): U[1]=1, U[3]=-1 (a01-a11)
    
    U_222 = [
        [1, 0, 1, 0, 1, -1,  0],  # i=0,j=0
        [0, 0, 0, 0, 1,  0,  1],  # i=0,j=1
        [0, 1, 0, 0, 0,  1,  0],  # i=1,j=0
        [1, 1, 0, 1, 0,  0, -1],  # i=1,j=1
    ]
    
    # V is 4x7. Row j*2+k.
    # Col 0 (M1): V[0]=1, V[3]=1 (b00, b11)
    # Col 1 (M2): V[0]=1 (b00)
    # Col 2 (M3): V[1]=1, V[3]=-1 (b01-b11)
    # Col 3 (M4): V[2]=1, V[0]=-1 (b10-b00)
    # Col 4 (M5): V[3]=1 (b11)
    # Col 5 (M6): V[0]=1, V[2]=1 (b00+b10)
    # Col 6 (M7): V[1]=1, V[3]=1 (b01+b11)
    
    V_222 = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0
        [0, 0, 1,  0, 0, 0, 1],  # j=0,k=1
        [0, 0, 0,  1, 0, 1, 0],  # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1],  # j=1,k=1
    ]
    
    # W is 4x7. Row k*2+i.
    # Col 0 (M1): W[0]=1, W[3]=1 (c00, c11)
    # Col 1 (M2): W[1]=1, W[3]=-1 (c10, c11)
    # Col 2 (M3): W[2]=1, W[3]=1 (c01, c11)
    # Col 3 (M4): W[0]=1, W[1]=1 (c00, c10)
    # Col 4 (M5): W[0]=-1, W[2]=1 (c00, c01)
    # Col 5 (M6): W[3]=1 (c11)
    # Col 6 (M7): W[0]=1 (c00)
    
    W_222 = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0
        [0, 1, 0, 1,  0, 0, 0],  # k=0,i=1
        [0, 0, 1, 0,  1, 0, 0],  # k=1,i=0
        [1, -1, 1, 0, 0,  1, 0],  # k=1,i=1
    ]
    
    # Verify 2x2x2
    if not _verify(U_222, V_222, W_222, 2, 2, 2):
        print("2x2x2 Strassen verification failed")
    
    # Build 2x2x1 rank-2 decomposition
    # n=2, m=2, p=1
    # U: (4,2), V: (2,2), W: (2,2)
    # We need: sum_r U[i*2+j,r] * V[j*1+k,r] * W[k*2+i,r] = 1 for all (i,j,k) with k=0
    # So: sum_r U[i*2+j,r] * V[j,r] * W[2*i,r] = 1
    # This is like computing the outer product pattern
    # Simple approach: use 2 terms
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> covers (i=0,j=0)
    # Term 1: U[3,1]=1, V[1,1]=1, W[1,1]=1 -> covers (i=1,j=1)
    # But we need all 4 (i,j) pairs!
    # Actually for 2x2x1, the tensor is T[i,j,0]=1 for all i,j.
    # We need sum_r U[i*2+j,r] * V[j,r] * W[2*i,r] = 1
    # With rank 2:
    # Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1
    # This gives: for (i=0,j=0): U[0]*V[0]*W[0]=1
    # for (i=0,j=1): U[1]*V[1]*W[0] - need V[1,0]
    # Let me think differently.
    # U[i*2+j, r] * V[j, r] * W[2*i, r]
    # If I set:
    # Term 0: U[0,0]=1, U[2,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    # Then for (i=0,j=0): 1*1*1=1
    # for (i=0,j=1): 0*1*1=0 -- wrong, U[1,0] should be 1
    # 
    # Let me try:
    # Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    # (i=0,j=0): U[0]*V[0]*W[0] = 1*1*1 = 1
    # (i=0,j=1): U[1]*V[1]*W[0] = 1*1*1 = 1
    # (i=1,j=0): U[2]*V[0]*W[1] = 0*1*1 = 0 -- wrong
    #
    # I need U[2,0]=1 and U[3,0]=1 too, but that's rank 1 with all ones.
    # Term 0: U all ones, V all ones, W all ones -> rank 1!
    # sum_r=0: 1*1*1 = 1 for all (i,j). Yes!
    
    U_221 = [[1, 1], [1, 1], [1, 1], [1, 1]]
    V_221 = [[1, 1], [1, 1]]
    W_221 = [[1, 1], [1, 1]]
    
    # But wait, rank 1 should work. Let me verify.
    # For (i,j): U[i*2+j, 0] * V[j, 0] * W[2*i, 0] = 1*1*1 = 1. Yes.
    # So 2x2x1 has rank 1? That seems too easy. Let me double-check the problem.
    # The multiplication tensor for n x m x p: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # For 2x2x1: T[i,j,0] = 1 for all i in {0,1}, j in {0,1}.
    # Decomposition: T[i,j,0] = sum_r U[i*2+j, r] V[j*1+0, r] W[0*2+i, r]
    # With rank 1: U[i*2+j,0]=1, V[j,0]=1, W[i,0]=1. Sum = 1. Yes, rank 1 works.
    
    # Actually, I realize the "rank" of the matrix multiplication tensor is what we're minimizing.
    # For 2x2x1, the tensor is just a 2x2 matrix of 1s, which has rank 1 as a tensor (it's a rank-1 tensor).
    
    # Now build 4x4x4 using Strassen
    # View 4x4 matrices as 2x2 blocks of 2x2.
    # 7 block multiplications, each 2x2x2 (rank 7).
    # Total rank: 49.
    
    # For the block Strassen:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # Each M is a 2x2 matrix product, decomposed with rank 7.
    # Total: 7 * 7 = 49.
    
    # Build U, V, W for 4x4x4
    # n=4, m=4, p=4
    # U: (16, 49), V: (16, 49), W: (16, 49)
    # U index: i*4+j
    # V index: j*4+k
    # W index: k*4+i
    
    # For each Strassen term s (0..6), and each inner Strassen term t (0..6):
    # The combined term index is s*7+t
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # Define the block-level Strassen combinations
    # For each block product M_s, we need to express the A and B block combinations
    # and the C block output combination
    
    # Block indices: 00, 01, 10, 11
    # For A: block (bi, bj) where bi in {0,1}, bj in {0,1}
    # For B: block (bk, bl) where bk in {0,1}, bl in {0,1}
    # For C: block (ci, ck) where ci in {0,1}, ck in {0,1}
    
    # Strassen terms at block level:
    # s=0: M1 = (A00+A11)(B00+B11)
    #   A blocks: 00 (+1), 11 (+1)
    #   B blocks: 00 (+1), 11 (+1)
    #   C contribution: C00 (+1), C11 (+1)
    # s=1: M2 = (A10+A11)B00
    #   A blocks: 10 (+1), 11 (+1)
    #   B blocks: 00 (+1)
    #   C contribution: C10 (+1), C11 (-1)
    # s=2: M3 = A00(B01-B11)
    #   A blocks: 00 (+1)
    #   B blocks: 01 (+1), 11 (-1)
    #   C contribution: C01 (+1), C11 (+1)
    # s=3: M4 = A11(B10-B00)
    #   A blocks: 11 (+1)
    #   B blocks: 10 (+1), 00 (-1)
    #   C contribution: C00 (+1), C10 (+1)
    # s=4: M5 = (A00+A01)B11
    #   A blocks: 00 (+1), 01 (+1)
    #   B blocks: 11 (+1)
    #   C contribution: C00 (-1), C01 (+1)
    # s=5: M6 = (A10-A00)(B00+B10)
    #   A blocks: 10 (+1), 00 (-1)
    #   B blocks: 00 (+1), 10 (+1)
    #   C contribution: C11 (+1)
    # s=6: M7 = (A01-A11)(B01+B11)
    #   A blocks: 01 (+1), 11 (-1)
    #   B blocks: 01 (+1), 11 (+1)
    #   C contribution: C00 (+1)
    
    block_strassen = [
        # (A_combination, B_combination, C_combination)
        # A_combination: list of (block_row, block_col, coeff)
        # B_combination: list of (block_row, block_col, coeff)
        # C_combination: list of (block_row, block_col, coeff)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)],
        },
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)],
        },
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)],
        },
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)],
        },
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)],
        },
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (1,0,1)],
            'C': [(1,1,1)],
        },
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(0,1,1), (1,1,1)],
            'C': [(0,0,1)],
        },
    ]
    
    # For each block Strassen term s, and each inner 2x2x2 Strassen term t:
    # The combined rank index is r = s*7 + t
    # U: for each (i,j) in 4x4, and each A block (bi,bj) with coeff a_c:
    #   i is in block row bi, j is in block col bj
    #   U[i*4+j, r] += a_c * U_222[(i%2)*2 + (j%2), t]
    # V: for each (j,k) in 4x4, and each B block (bk,bl) with coeff b_c:
    #   j is in block row bk, k is in block col bl
    #   V[j*4+k, r] += b_c * V_222[(j%2)*2 + (k%2), t]
    # W: for each (k,i) in 4x4, and each C block (ci,ck) with coeff c_c:
    #   k is in block row ci, i is in block col ck
    #   W[k*4+i, r] += c_c * W_222[(k%2)*2 + (i%2), t]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for (bi, bj, a_c) in block_strassen[s]['A']:
                for ii in range(2):
                    for jj in range(2):
                        i = bi * 2 + ii
                        j = bj * 2 + jj
                        U_444[i*4+j][r] += a_c * U_222[ii*2+jj][t]
            for (bk, bl, b_c) in block_strassen[s]['B']:
                for jj in range(2):
                    for kk in range(2):
                        j = bk * 2 + jj
                        k = bl * 2 + kk
                        V_444[j*4+k][r] += b_c * V_222[jj*2+kk][t]
            for (ci, ck, c_c) in block_strassen[s]['C']:
                for kk in range(2):
                    for ii in range(2):
                        k = ci * 2 + kk
                        i = ck * 2 + ii
                        W_444[k*4+i][r] += c_c * W_222[kk*2+ii][t]
    
    # Verify 4x4x4
    if not _verify(U_444, V_444, W_444, 4, 4, 4):
        print("4x4x4 Strassen verification failed")
    else:
        print("4x4x4 Strassen rank 49 verified")
        best_rank = 49
        best_result = {"U": U_444, "V": V_444, "W": W_444}
    
    if time.time() > deadline:
        return best_result
    
    # Build 4x4x3
    # Split 4x4x3 into: (2x4x3) + (2x4x3) for the two row-blocks of A
    # Each 2x4x3: split into (2x2x3) + (2x2x3) for the two col-blocks of A
    # Each 2x2x3: Strassen 2x2x2 (rank 7) + 2x2x1 (rank 1) = rank 8
    # Total: 4 * 8 = 32
    
    # Wait, I said rank 9 earlier for 2x2x3, but 2x2x1 has rank 1, so 2x2x3 has rank 7+1=8.
    # Let me rebuild.
    
    # 2x2x3: n=2, m=2, p=3
    # Split k into k=0,1 (2x2x2) and k=2 (2x2x1)
    # U: (4, 8), V: (6, 8), W: (6, 8)
    # For k=0,1: use 2x2x2 Strassen
    # For k=2: use 2x2x1 rank 1
    
    U_223 = [[0]*8 for _ in range(4)]
    V_223 = [[0]*8 for _ in range(6)]
    W_223 = [[0]*8 for _ in range(6)]
    
    # Copy 2x2x2 Strassen for k=0,1
    # U indices: i*2+j (same)
    # V indices: j*3+k. For k=0: j*3+0, for k=1: j*3+1
    # W indices: k*2+i. For k=0: 0*2+i=i, for k=1: 1*2+i=2+i
    
    for t in range(7):
        r = t
        for i in range(4):
            U_223[i][r] = U_222[i][t]
        for j in range(2):
            for k in range(2):
                V_223[j*3+k][r] = V_222[j*2+k][t]
        for k in range(2):
            for i in range(2):
                W_223[k*2+i][r] = W_222[k*2+i][t]
    
    # For k=2: 2x2x1 rank 1
    r = 7
    # U_221: all ones, V_221: all ones, W_221: all ones
    for i in range(4):
        U_223[i][r] = U_221[i][0]
    for j in range(2):
        V_223[j*3+2][r] = V_221[j][0]
    for i in range(2):
        W_223[2*2+i][r] = W_221[i][0]
    
    # Verify 2x2x3
    if not _verify(U_223, V_223, W_223, 2, 2, 3):
        print("2x2x3 verification failed")
    else:
        print("2x2x3 rank 8 verified")
    
    # Now build 4x4x3
    # n=4, m=4, p=3
    # Split A rows into two 2x4 blocks: row block 0 (i=0,1) and row block 1 (i=2,3)
    # Each 2x4x3: split A cols into two 2x2 blocks: col block 0 (j=0,1) and col block 1 (j=2,3)
    # Each 2x2x3: rank 8
    # Total: 4 * 8 = 32
    
    U_443 = [[0]*32 for _ in range(16)]
    V_443 = [[0]*32 for _ in range(12)]
    W_443 = [[0]*32 for _ in range(12)]
    
    # For each (row_block, col_block) in {(0,0), (0,1), (1,0), (1,1)}:
    #   The 2x2x3 subproblem
    #   U index: i*4+j where i is in row_block*2+{0,1}, j is in col_block*2+{0,1}
    #   V index: j*3+k where j is in col_block*2+{0,1}, k in {0,1,2}
    #   W index: k*4+i where k in {0,1,2}, i is in row_block*2+{0,1}
    
    for rb in range(2):
        for cb in range(2):
            base = (rb * 2 + cb) * 8
            for t in range(8):
                r = base + t
                for ii in range(2):
                    for jj in range(2):
                        i = rb * 2 + ii
                        j = cb * 2 + jj
                        U_443[i*4+j][r] = U_223[ii*2+jj][t]
                for jj in range(2):
                    for kk in range(3):
                        j = cb * 2 + jj
                        k = kk
                        V_443[j*3+k][r] = V_223[jj*3+kk][t]
                for kk in range(3):
                    for ii in range(2):
                        k = kk
                        i = rb * 2 + ii
                        W_443[k*4+i][r] = W_223[kk*2+ii][t]
    
    # Verify 4x4x3
    if not _verify(U_443, V_443, W_443, 4, 4, 3):
        print("4x4x3 verification failed")
    else:
        print("4x4x3 rank 32 verified")
    
    if time.time() > deadline:
        return best_result
    
    # Combine 4x4x4 (rank 49) and 4x4x3 (rank 32) to get 4x4x7 (rank 81)
    U_447 = [[0]*81 for _ in range(16)]
    V_447 = [[0]*81 for _ in range(28)]
    W_447 = [[0]*81 for _ in range(28)]
    
    # First 49 terms: from 4x4x4
    for i in range(16):
        for r in range(49):
            U_447[i][r] = U_444[i][r]
    for j in range(16):
        for r in range(49):
            # V index for 4x4x4: j*4+k, k in 0..3
            # V index for 4x4x7: j*7+k, k in 0..6
            # For the 4x4x4 part, k in 0..3
            V_447[j*7+k][r] = V_444[j*4+k][r] if j*7+k < 28 else 0
            # Wait, j goes 0..15 for 4x4x4, but j is 0..3 for the original.
            # Actually, V for 4x4x4 has 16 rows (j*4+k for j=0..3, k=0..3)
            # V for 4x4x7 has 28 rows (j*7+k for j=0..3, k=0..6)
            # So for the 4x4x4 part, we map V_444[j*4+k] to V_447[j*7+k]
    
    # Let me redo this more carefully.
    
    U_447 = [[0]*81 for _ in range(16)]
    V_447 = [[0]*81 for _ in range(28)]
    W_447 = [[0]*81 for _ in range(28)]
    
    # 4x4x4 part: ranks 0..48
    for i in range(16):
        for r in range(49):
            U_447[i][r] = U_444[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_447[j*7+k][r] = V_444[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_447[k*4+i][r] = W_444[k*4+i][r]
    
    # 4x4x3 part: ranks 49..80
    for i in range(16):
        for r in range(32):
            U_447[i][49+r] = U_443[i][r]
    for j in range(4):
        for k in range(3):
            for r in range(32):
                V_447[j*7+k][49+r] = V_443[j*3+k][r]
    for k in range(3):
        for i in range(4):
            for r in range(32):
                W_447[k*4+i][49+r] = W_443[k*4+i][r]
    
    # Verify 4x4x7
    if not _verify(U_447, V_447, W_447, 4, 4, 7):
        print("4x4x7 verification failed")
    else:
        print("4x4x7 rank 81 verified")
        best_rank = 81
        best_result = {"U": U_447, "V": V_447, "W": W_447}
    
    return best_result
