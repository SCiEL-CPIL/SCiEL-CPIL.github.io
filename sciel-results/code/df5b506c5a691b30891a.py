import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (Rank 7).
    Indices:
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1  # C00
    W[3,0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1  # C10
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1  # C01
    W[2,2] = 1  # C10
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = -1  # C00
    W[1,3] = 1   # C01
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1  # C00
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1  # C00
    W[3,6] = -1  # C11
    
    return U, V, W

def _build_4x4x4_strassen_block(col_offset):
    """
    Builds the factors for a 4x4x4 matrix multiplication using 8 Strassen 2x2x2 blocks.
    Returns the contributions to U, V, W starting at col_offset.
    Rank used: 8 * 7 = 56.
    """
    n, m, p = 4, 4, 4
    rank_block = 56
    U = np.zeros((n*m, rank_block), dtype=int)
    V = np.zeros((m*p, rank_block), dtype=int)
    W = np.zeros((p*n, rank_block), dtype=int)
    
    U_s, V_s, W_s = _get_strassen_2x2x2()
    
    # 8 blocks corresponding to (i_block, j_block, k_block)
    # A is 4x4, partitioned into 2x2 blocks (bi, bj)
    # B is 4x4, partitioned into 2x2 blocks (bj, bk)
    # C is 4x4, partitioned into 2x2 blocks (bk, bi)
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # This block computes C_bk,bi = A_bi,bj * B_bj,bk
                # Wait, the standard block multiplication is:
                # C_{ab} = sum_c A_{ac} B_{cb}
                # So C_{bk, bi} depends on sum_{bj} A_{bi, bj} B_{bj, bk}
                # There are 8 terms in the sum? No, for 4x4, there are 2 terms in the sum.
                # C_{bk, bi} = A_{bi, 0} B_{0, bk} + A_{bi, 1} B_{1, bk}
                # So for each output block (bk, bi), we have 2 input blocks.
                # My previous logic of 8 blocks was for the 8 products A_ij B_jk.
                # Yes, there are 8 products of 2x2 matrices.
                
                # We add the Strassen decomposition for A_{bi,bj} * B_{bj,bk}
                # to the result C_{bk,bi}.
                
                # U indices: A is 4x4. Block (bi, bj) is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
                # U row index: i*m + j.
                # For local i in 0..1, j in 0..1:
                # global i = 2*bi + i, global j = 2*bj + j.
                # global index = (2*bi+i)*4 + (2*bj+j).
                
                # V indices: B is 4x4. Block (bj, bk) is rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
                # V row index: j*p + k.
                # For local j in 0..1, k in 0..1:
                # global j = 2*bj + j, global k = 2*bk + k.
                # global index = (2*bj+j)*4 + (2*bk+k).
                
                # W indices: C is 4x4. Block (bk, bi) is rows 2*bk..2*bk+1, cols 2*bi..2*bi+1.
                # W row index: k*n + i.
                # For local k in 0..1, i in 0..1:
                # global k = 2*bk + k, global i = 2*bi + i.
                # global index = (2*bk+k)*4 + (2*bi+i).
                
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            u_idx = (2*bi + i) * 4 + (2*bj + j)
                            U[u_idx, col] = U_s[i*2+j, r]
                    
                    for j in range(2):
                        for k in range(2):
                            v_idx = (2*bj + j) * 4 + (2*bk + k)
                            V[v_idx, col] = V_s[j*2+k, r]
                            
                    for k in range(2):
                        for i in range(2):
                            w_idx = (2*bk + k) * 4 + (2*bi + i)
                            W[w_idx, col] = W_s[k*2+i, r]
                            
                    col += 1
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{4,4,9} into two M_{4,4,4} and one M_{4,4,1}
    # Rank 1: 56 (from 8 Strassen blocks)
    # Rank 2: 56 (from 8 Strassen blocks)
    # Rank 3: 16 (Schoolbook for 4x4x1)
    # Total Rank: 128
    
    rank = 128
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col_offset = 0
    
    # Block 1: k in 0..3
    U1, V1, W1 = _build_4x4x4_strassen_block(col_offset)
    U[:, col_offset:col_offset+56] = U1
    # V1 is 4x4, we need to map to 4x9.
    # V1 indices are j*4+k. We need j*9+k.
    V_block1 = np.zeros((m*p, 56), dtype=int)
    for j in range(m):
        for k in range(4):
            V_block1[j*p+k, :] = V1[j*4+k, :]
    V[:, col_offset:col_offset+56] = V_block1
    
    # W1 is 4x4, we need to map to 9x4.
    # W1 indices are k*4+i. We need k*4+i (since p=9, n=4, index is k*n+i).
    # Wait, W shape is (p*n, rank). Index is k*n+i.
    # In the 4x4x4 block, p=4, n=4. Index is k*4+i.
    # In the 4x4x9 block, p=9, n=4. Index is k*4+i.
    # So the indices match for k in 0..3.
    W[:, col_offset:col_offset+56] = W1
    
    col_offset += 56
    
    # Block 2: k in 4..7
    U2, V2, W2 = _build_4x4x4_strassen_block(col_offset)
    U[:, col_offset:col_offset+56] = U2
    V_block2 = np.zeros((m*p, 56), dtype=int)
    for j in range(m):
        for k in range(4):
            # k in V2 is 0..3. Global k is 4+k.
            V_block2[j*p + (4+k), :] = V2[j*4+k, :]
    V[:, col_offset:col_offset+56] = V_block2
    
    # W2 indices are k*4+i. Global k is 4+k.
    # W index is global_k * n + i = (4+k)*4 + i = 16 + 4k + i.
    W_block2 = np.zeros((p*n, 56), dtype=int)
    for k in range(4):
        for i in range(n):
            W_block2[(4+k)*n + i, :] = W2[k*4+i, :]
    W[:, col_offset:col_offset+56] = W_block2
    
    col_offset += 56
    
    # Block 3: k in 8..8 (4x4x1)
    # Schoolbook for 4x4x1. Rank 16.
    for i in range(n):
        for j in range(m):
            U[i*m+j, col_offset] = 1
            V[j*p + 8, col_offset] = 1
            W[8*n + i, col_offset] = 1
            col_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
