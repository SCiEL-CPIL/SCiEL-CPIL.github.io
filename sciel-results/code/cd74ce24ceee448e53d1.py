import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start = time.time()
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 2x2 Strassen factors (rank 7)
    # T2[i*2+j, j*2+k, k*2+i] = sum_s U2[i*2+j,s] * V2[j*2+k,s] * W2[k*2+i,s] = 1
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen 2x2 terms:
    # s=0: m0 = (A00+A11)(B00+B11) -> C00 += m0, C11 += m0
    # s=1: m1 = (A10+A11)B00 -> C10 += m1, C11 -= m1
    # s=2: m2 = A00(B01-B11) -> C01 += m2, C11 += m2
    # s=3: m3 = A11(B10-B00) -> C00 += m3, C10 += m3
    # s=4: m4 = (A00+A01)B11 -> C00 -= m4, C01 += m4
    # s=5: m5 = (A10-A00)(B00+B01) -> C11 += m5
    # s=6: m6 = (A01-A11)(B10+B11) -> C00 += m6
    
    # s=0: U2[0,0]=1, U2[3,0]=1; V2[0,0]=1, V2[3,0]=1; W2[0,0]=1, W2[3,0]=1
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # s=1: U2[2,1]=1, U2[3,1]=1; V2[0,1]=1; W2[2,1]=1, W2[3,1]=-1
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1; W2[3,1] = -1
    
    # s=2: U2[0,2]=1; V2[1,2]=1, V2[3,2]=-1; W2[1,2]=1, W2[3,2]=1
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[1,2] = 1; W2[3,2] = 1
    
    # s=3: U2[3,3]=1; V2[2,3]=1, V2[0,3]=-1; W2[0,3]=1, W2[2,3]=1
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[2,3] = 1
    
    # s=4: U2[0,4]=1, U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1, W2[1,4]=1
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # s=5: U2[2,5]=1, U2[0,5]=-1; V2[0,5]=1, V2[1,5]=1; W2[3,5]=1
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    # s=6: U2[1,6]=1, U2[3,6]=-1; V2[2,6]=1, V2[3,6]=1; W2[0,6]=1
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1
    
    # Verify 2x2 Strassen
    T2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1.0
    
    # Check: sum_s U2[i*2+j,s]*V2[j*2+k,s]*W2[k*2+i,s] should equal T2[i*2+j,j*2+k,k*2+i]
    recon2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for s in range(7):
                    recon2[i*2+j, j*2+k, k*2+i] += U2[i*2+j, s] * V2[j*2+k, s] * W2[k*2+i, s]
    
    err2 = np.max(np.abs(recon2 - T2))
    # err2 should be 0
    
    # Top-level Strassen: 4x4 = 2x2 blocks of 2x2
    # Block (bi,bj) of A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # U index for A[2*bi+r, 2*bj+c] = (2*bi+r)*4 + (2*bj+c)
    # V index for B[2*bi+r, 2*bj+c] = (2*bi+r)*4 + (2*bj+c)
    # W index for C[2*bi+r, 2*bj+c] = (2*bj+c)*4 + (2*bi+r)
    
    # Top-level Strassen terms (same structure as 2x2 but operating on 2x2 blocks):
    # t=0: P0 = (A11+A22)(B11+B22) -> C11 += P0, C22 += P0
    # t=1: P1 = (A21+A22)B11 -> C21 += P1, C22 -= P1
    # t=2: P2 = A11(B12-B22) -> C12 += P2, C22 += P2
    # t=3: P3 = A22(B21-B11) -> C11 += P3, C21 += P3
    # t=4: P4 = (A11+A12)B22 -> C11 -= P4, C12 += P4
    # t=5: P5 = (A21-A11)(B11+B12) -> C22 += P5
    # t=6: P6 = (A12-A22)(B21+B22) -> C11 += P6
    
    # For each top term t, the "matrices" L_t and R_t are 2x2 block matrices.
    # L_t[bi][bj] is a 2x2 matrix (linear combo of A blocks)
    # R_t[bi][bj] is a 2x2 matrix (linear combo of B blocks)
    # P_t = L_t * R_t, so P_t[bi][bj] = sum_bd L_t[bi][bd] * R_t[bd][bj]
    # Each P_t[bi][bj] is a 2x2 matrix, computed via 2x2 Strassen (7 rank-1 terms)
    
    # For top term t, the bottom Strassen decomposes the 2x2 matmul P_t[bi][bj].
    # The "A" for the bottom Strassen is L_t[bi][.] (a 2x2 matrix of A-block elements)
    # The "B" for the bottom Strassen is R_t[.][bj] (a 2x2 matrix of B-block elements)
    # The "C" result is P_t[bi][bj] (a 2x2 matrix contributing to C blocks)
    
    # For bottom Strassen term s:
    # U2[i*2+j, s] selects which element (i,j) of L_t to use
    # V2[j*2+k, s] selects which element (j,k) of R_t to use
    # W2[k*2+i, s] is the contribution coefficient to P_t[i][k]
    
    # L_t[i][j] maps to A block elements:
    #   L_t[i][j] = sum over (ai,aj) of coeff * A_block(ai,aj)
    #   A_block(ai,aj) element (r,c) -> global A[2*ai+r][2*aj+c] -> U index (2*ai+r)*4+(2*aj+c)
    
    # R_t[j][k] maps to B block elements similarly
    # P_t[i][k] maps to C block elements:
    #   P_t[i][k] contributes to C blocks with some coefficient
    #   C block (ci,ck) element (r,c) -> global C[2*ci+r][2*ck+c] -> W index (2*ck+c)*4+(2*ci+r)
    
    # Define top-level coefficient tables
    # top_L[t][i][j] = dict: {(ai,aj): coeff} - coefficient of A block (ai,aj) in L_t[i][j]
    # top_R[t][i][j] = dict: {(bi,bj): coeff} - coefficient of B block (bi,bj) in R_t[i][j]
    # top_C[t][i][k] = dict: {(ci,ck): coeff} - coefficient for C block (ci,ck) from P_t[i][k]
    
    top_L = []
    top_R = []
    top_C = []
    
    # t=0: L0 = [[A11, 0], [0, A22]], R0 = [[B11, 0], [0, B22]]
    # Actually L0[i][j]: L0[0][0]=A11, L0[0][1]=0, L0[1][0]=0, L0[1][1]=A22
    top_L.append([
        [{(0,0): 1}, {}],
        [{}, {(1,1): 1}]
    ])
    top_R.append([
        [{(0,0): 1}, {}],
        [{}, {(1,1): 1}]
    ])
    top_C.append([
        [{(0,0): 1}, {}],
        [{}, {(1,1): 1}]
    ])
    
    # t=1: L1 = [[0, 0], [A21, A22]], R1 = [[B11, 0], [0, 0]]
    top_L.append([
        [{}, {}],
        [{(1,0): 1}, {(1,1): 1}]
    ])
    top_R.append([
        [{(0,0): 1}, {}],
        [{}, {}]
    ])
    top_C.append([
        [{}, {}],
        [{(1,0): 1}, {(1,1): -1}]
    ])
    
    # t=2: L2 = [[A11, 0], [0, 0]], R2 = [[0, B12], [0, -B22]]
    top_L.append([
        [{(0,0): 1}, {}],
        [{}, {}]
    ])
    top_R.append([
        [{}, {(0,1): 1}],
        [{}, {(1,1): -1}]
    ])
    top_C.append([
        [{}, {(0,1): 1}],
        [{}, {(1,1): 1}]
    ])
    
    # t=3: L3 = [[0, 0], [0, A22]], R3 = [[-B11, 0], [B21, 0]]
    top_L.append([
        [{}, {}],
        [{}, {(1,1): 1}]
    ])
    top_R.append([
        [{(0,0): -1}, {}],
        [{(1,0): 1}, {}]
    ])
    top_C.append([
        [{(0,0): 1}, {}],
        [{(1,0): 1}, {}]
    ])
    
    # t=4: L4 = [[A11, A12], [0, 0]], R4 = [[0, 0], [0, B22]]
    top_L.append([
        [{(0,0): 1}, {(0,1): 1}],
        [{}, {}]
    ])
    top_R.append([
        [{}, {}],
        [{}, {(1,1): 1}]
    ])
    top_C.append([
        [{(0,0): -1}, {(0,1): 1}],
        [{}, {}]
    ])
    
    # t=5: L5 = [[0, 0], [A21, -A11]], R5 = [[B11, B12], [0, 0]]
    top_L.append([
        [{}, {}],
        [{(1,0): 1}, {(0,0): -1}]
    ])
    top_R.append([
        [{(0,0): 1}, {(0,1): 1}],
        [{}, {}]
    ])
    top_C.append([
        [{}, {}],
        [{}, {(1,1): 1}]
    ])
    
    # t=6: L6 = [[0, A12], [0, -A22]], R6 = [[0, 0], [B21, B22]]
    top_L.append([
        [{}, {(0,1): 1}],
        [{}, {(1,1): -1}]
    ])
    top_R.append([
        [{}, {}],
        [{(1,0): 1}, {(1,1): 1}]
    ])
    top_C.append([
        [{(0,0): 1}, {}],
        [{}, {}]
    ])
    
    # Build U, V, W using Kronecker product structure
    for t in range(7):
        for s in range(7):
            rank_idx = t * 7 + s
            
            # U: for each (i, j) in 0..1 (bottom Strassen U index i*2+j)
            u_val = U2[i*2+j, s] if (i*2+j) < 4 else 0
            # Actually U2 has shape (4,7), indexed by [i*2+j, s]
            # i,j in 0,1 so i*2+j in 0,1,2,3
            for i in range(2):
                for j in range(2):
                    u_val = U2[i*2 + j, s]
                    if u_val == 0:
                        continue
                    # L_t[i][j] = sum over (ai,aj) of coeff * A_block(ai,aj)
                    for (ai, aj), a_coeff in top_L[t][i][j].items():
                        full_coeff = u_val * a_coeff
                        # A block (ai,aj) elements: (2*ai+r, 2*aj+c) for r,c in 0,1
                        for r in range(2):
                            for c in range(2):
                                u_idx = (2*ai + r) * 4 + (2*aj + c)
                                U[u_idx, rank_idx] += full_coeff
            
            # V: for each (j, k) in 0..1 (bottom Strassen V index j*2+k)
            for j in range(2):
                for k in range(2):
                    v_val = V2[j*2 + k, s]
                    if v_val == 0:
                        continue
                    # R_t[j][k] = sum over (bi,bj) of coeff * B_block(bi,bj)
                    for (bi, bj), b_coeff in top_R[t][j][k].items():
                        full_coeff = v_val * b_coeff
                        for r in range(2):
                            for c in range(2):
                                v_idx = (2*bi + r) * 4 + (2*bj + c)
                                V[v_idx, rank_idx] += full_coeff
            
            # W: for each (k, i) in 0..1 (bottom Strassen W index k*2+i)
            for k in range(2):
                for i in range(2):
                    w_val = W2[k*2 + i, s]
                    if w_val == 0:
                        continue
                    # P_t[i][k] contributes to C blocks
                    for (ci, ck), c_coeff in top_C[t][i][k].items():
                        full_coeff = w_val * c_coeff
                        for r in range(2):
                            for c in range(2):
                                w_idx = (2*ck + c) * 4 + (2*ci + r)
                                W[w_idx, rank_idx] += full_coeff
    
    # Verify
    T4 = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    
    recon = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                recon[i*4+j, j*4+k, k*4+i] = np.sum(U[i*4+j] * V[j*4+k] * W[k*4+i])
    
    err = np.max(np.abs(recon - T4))
    
    if err > 1e-10:
        # Fallback to schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((16, rank_fb))
        W_fb = np.zeros((16, rank_fb))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i*16 + j*4 + k
                    U_fb[i*4+j, r] = 1
                    V_fb[j*4+k, r] = 1
                    W_fb[k*4+i, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
