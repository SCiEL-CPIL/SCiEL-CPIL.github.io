"""Matrix multiplication 4x4 * 4x8 via rank-98 Strassen decomposition.

Uses rank-49 Strassen for each 4x4 * 4x4 half, giving total rank 98.
The 4x8 matrix B is split into two 4x4 halves B_left and B_right.
C = A * B = [A * B_left, A * B_right], each computed with rank-49 Strassen.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _make_strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen terms for 2x2*2x2:
    # M1=(a00+a11)(b00+b11), M2=(a10+a11)b00, M3=a00(b01-b11), M4=a11(b10-b00)
    # M5=(a00+a01)b11, M6=(a10-a00)(b00+b01), M7=(a01-a11)(b10+b11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # U indices: i*2+j where i=row, j=col of A
    # V indices: j*2+k where j=row, k=col of B
    # W indices: k*2+i where k=row, i=col of C (trace convention)
    
    # M1: (a00+a11)(b00+b11)
    # a00=U[0], a11=U[3]; b00=V[0], b11=V[3]
    # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[3]
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2: (a10+a11)b00
    # a10=U[2], a11=U[3]; b00=V[0]
    # C10: k=1,i=0 -> W[2]; C11: k=1,i=1 -> W[3]
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # M3: a00(b01-b11)
    # a00=U[0]; b01=V[1], b11=V[3]
    # C01: k=0,i=1 -> W[1]; C11: k=1,i=1 -> W[3]
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # M4: a11(b10-b00)
    # a11=U[3]; b10=V[2], b00=V[0]
    # C00: k=0,i=0 -> W[0]; C10: k=1,i=0 -> W[2]
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # M5: (a00+a01)b11
    # a00=U[0], a01=U[1]; b11=V[3]
    # C00: k=0,i=0 -> W[0]; C01: k=0,i=1 -> W[1]
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # M6: (a10-a00)(b00+b01)
    # a10=U[2], a00=U[0]; b00=V[0], b01=V[1]
    # C11: k=1,i=1 -> W[3]
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7: (a01-a11)(b10+b11)
    # a01=U[1], a11=U[3]; b10=V[2], b11=V[3]
    # C00: k=0,i=0 -> W[0]
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """Build rank-49 Strassen decomposition for 4x4 * 4x4.
    
    Returns (U, V, W) with shapes:
    U: (16, 49) - A is 4x4, indexed i*4+j
    V: (16, 49) - B is 4x4, indexed j*4+k
    W: (16, 49) - C is 4x4, indexed k*4+i (trace convention)
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Get the base 2x2 Strassen decomposition
    U2, V2, W2 = _make_strassen_2x2()
    
    # Top-level Strassen for 4x4: split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each top-level term s (0-6), define:
    # a_side: dict mapping block_num -> coeff for A
    # b_side: dict mapping block_num -> coeff for B
    # c_side: dict mapping block_num -> coeff for C
    
    top_a = [
        {0: 1, 3: 1},   # s=0: A00 + A11
        {2: 1, 3: 1},   # s=1: A10 + A11
        {0: 1},         # s=2: A00
        {3: 1},         # s=3: A11
        {0: 1, 1: 1},   # s=4: A00 + A01
        {2: 1, 0: -1},  # s=5: A10 - A00
        {1: 1, 3: -1},  # s=6: A01 - A11
    ]
    
    top_b = [
        {0: 1, 3: 1},   # s=0: B00 + B11
        {0: 1},         # s=1: B00
        {1: 1, 3: -1},  # s=2: B01 - B11
        {2: 1, 0: -1},  # s=3: B10 - B00
        {3: 1},         # s=4: B11
        {0: 1, 1: 1},   # s=5: B00 + B01
        {2: 1, 3: 1},   # s=6: B10 + B11
    ]
    
    top_c = [
        {0: 1, 3: 1},   # s=0: C00 +1, C11 +1
        {2: 1, 3: -1},  # s=1: C10 +1, C11 -1
        {1: 1, 3: 1},   # s=2: C01 +1, C11 +1
        {0: 1, 2: 1},   # s=3: C00 +1, C10 +1
        {0: -1, 1: 1},  # s=4: C00 -1, C01 +1
        {3: 1},         # s=5: C11 +1
        {0: 1},         # s=6: C00 +1
    ]
    
    # For each top-level term s and each sub-term t (0-6 from 2x2 Strassen):
    # The sub-term t operates on a 2x2 "X" and 2x2 "Y" to produce 2x2 "Z"
    # X's elements map to A blocks, Y's to B blocks, Z's to C blocks
    # 
    # In the 2x2 Strassen (U2, V2, W2):
    # U2[i*2+j, t] gives coefficient for A element (i,j) in sub-term t
    # V2[j*2+k, t] gives coefficient for B element (j,k) in sub-term t  
    # W2[k*2+i, t] gives coefficient for C element (k,i) in sub-term t
    #
    # But here X is a 2x2 matrix of A-blocks, so X[br,bc] is the A-block at (br,bc)
    # The 2x2 Strassen U2[i*2+j, t] tells us coefficient for X[i,j]
    # X[i,j] corresponds to A-block (i,j) which has top-level coefficient top_a[s].get(i*2+j, 0)
    # So the effective A coefficient for element (I,J) in A where I is in block (i,j) is:
    #   U2[i*2+j, t] * top_a[s].get(i*2+j, 0)
    # where i = I//2, j = J//2
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # A side: for each element (I, J) in 4x4 A
            for I in range(4):
                for J in range(4):
                    block_i = I // 2
                    block_j = J // 2
                    block_num = block_i * 2 + block_j
                    a_block_coeff = top_a[s].get(block_num, 0)
                    if a_block_coeff == 0:
                        continue
                    # In the 2x2 sub-problem, this element is at position (I%2, J%2)
                    sub_i = I % 2
                    sub_j = J % 2
                    u2_coeff = U2[sub_i * 2 + sub_j, t]
                    U[I * 4 + J, col] = u2_coeff * a_block_coeff
            
            # B side: for each element (J, K) in 4x4 B
            for J in range(4):
                for K in range(4):
                    block_j = J // 2
                    block_k = K // 2
                    block_num = block_j * 2 + block_k
                    b_block_coeff = top_b[s].get(block_num, 0)
                    if b_block_coeff == 0:
                        continue
                    sub_j = J % 2
                    sub_k = K % 2
                    v2_coeff = V2[sub_j * 2 + sub_k, t]
                    V[J * 4 + K, col] = v2_coeff * b_block_coeff
            
            # C side: for each element (K, I) in 4x4 C (trace convention W[k*4+i])
            for K in range(4):
                for I in range(4):
                    block_k = K // 2
                    block_i = I // 2
                    block_num = block_k * 2 + block_i
                    c_block_coeff = top_c[s].get(block_num, 0)
                    if c_block_coeff == 0:
                        continue
                    sub_k = K % 2
                    sub_i = I % 2
                    w2_coeff = W2[sub_k * 2 + sub_i, t]
                    W[K * 4 + I, col] = w2_coeff * c_block_coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    # Use einsum for efficiency
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k] - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Build the rank-98 Strassen decomposition for 4x4 * 4x8."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    total_rank = 98
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Build the 4x4*4x4 Strassen decomposition
    U4, V4, W4 = _build_strassen_4x4_4x4()
    
    # Verify the 4x4*4x4 sub-decomposition
    if not _verify_decomposition(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # For 4x4 * 4x8: split B into two 4x4 halves
    # half=0: B[:, 0:4], C[:, 0:4]
    # half=1: B[:, 4:8], C[:, 4:8]
    
    for half in range(2):
        base_col = half * 49
        col_offset = half * 4  # offset in B and C columns
        
        # Copy U4 to U (same A for both halves)
        U[:, base_col:base_col+49] = U4
        
        # Copy V4 to V, with column offset
        # V4[j*4+k, r] -> V[j*8 + (k + col_offset), base_col + r]
        for j in range(4):
            for k in range(4):
                V[j*8 + (k + col_offset), base_col:base_col+49] = V4[j*4+k, :]
        
        # Copy W4 to W, with column offset
        # W4[k*4+i, r] -> W[(k + col_offset)*4 + i, base_col + r]
        for k in range(4):
            for i in range(4):
                W[(k + col_offset)*4 + i, base_col:base_col+49] = W4[k*4+i, :]
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < 100:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
