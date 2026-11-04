"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_4x4x4():
    """
    Returns U, V, W for 4x4 x 4x4 matrix multiplication using Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    # Strassen for 2x2
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11) -> C21 += M4, C11 -= M4
    # M5 = (A11+A12)B22 -> C12 -= M5, C11 += M5
    # M6 = (A21-A11)(B11+B12) -> C21 += M6
    # M7 = (A12-A22)(B21+B22) -> C12 += M7
    
    # We build 4x4x4 by using 7 Strassen multiplications of 2x2 blocks.
    # There are 4 2x2 blocks in A, 4 in B, 4 in C.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 - P2  (Wait, standard Strassen: C00 = P1 - P4 + P5 - P2? No.)
    # Standard 2x2 Strassen:
    # C00 = M1 - M4 + M5
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3
    
    # Let's verify:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5?
    # M1 - M4 + M5 = (A00+A11)(B00+B11) - A11(B10-B00) + (A00+A01)B11
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B10 + A11B00 + A00B11 + A01B11
    # This doesn't look like A00B00 + A01B10.
    
    # Correct Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3
    
    # Let's map 4x4 blocks to 2x2 indices.
    # A00 is 2x2 block (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1).
    # In Strassen notation for the 2x2 "super" matrix:
    # A11_super = A00_4x4? No.
    # Let's just use the 7 products defined above and map them to C blocks.
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    # We will define the 7 "super" Strassen multiplications.
    # Each super multiplication is a 2x2 x 2x2 Strassen (rank 7).
    # Total rank 49.
    
    # Helper to add a 2x2 Strassen block
    def add_strassen_block(a_blocks, b_blocks, c_blocks, offset):
        # a_blocks: list of (block_idx, sign) for A
        # b_blocks: list of (block_idx, sign) for B
        # c_blocks: list of (block_idx, sign) for C
        # This is complex to generalize. Let's hardcode the 7 products.
        
        pass
        
    # Let's just build the 2x2 Strassen once and replicate.
    # U2, V2, W2 for 2x2x2 Strassen
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # M1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1
    
    # M3
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1
    
    # M4
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[1,3]=1
    
    # M5
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1
    
    # M6
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Now construct 4x4x4.
    # The 7 products P1..P7 are 2x2x2 multiplications of 2x2 blocks.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 - P4 + P5  (Check: A00B00 + A01B10? No, C00 is top-left 2x2 of C.
    # C00 = A00B00 + A01B10.
    # Let's check P1 - P4 + P5.
    # P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # P4 = A11(B10-B00) = A11B10 - A11B00
    # P5 = (A00+A01)B11 = A00B11 + A01B11
    # P1 - P4 + P5 = A00B00 + A00B11 + A11B00 + A11B11 - A11B10 + A11B00 + A00B11 + A01B11
    # This is not A00B00 + A01B10.
    
    # I need the correct Strassen mapping for 4x4.
    # The 4x4 matrix A is composed of 2x2 blocks A00, A01, A10, A11.
    # We treat A as a 2x2 matrix of blocks.
    # We apply Strassen to this 2x2 matrix of blocks.
    # The "entries" of this 2x2 matrix are 2x2 matrices.
    # So the multiplication is over the ring of 2x2 matrices.
    # Strassen's algorithm works over any commutative ring.
    # So we just apply the 7 Strassen formulas to the blocks.
    
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The result blocks are:
    # C00 = M1 - M4 + M5
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3
    
    # Let's verify C00 = A00B00 + A01B10.
    # M1 - M4 + M5
    # = (A00+A11)(B00+B11) - A11(B10-B00) + (A00+A01)B11
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B10 + A11B00 + A00B11 + A01B11
    # Still doesn't match.
    
    # Ah, the block indices in Strassen:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3
    
    # Map 4x4 blocks to 2x2 block indices:
    # A11_block = A00 (top-left 2x2 of 4x4)
    # A12_block = A01 (top-right 2x2 of 4x4)
    # A21_block = A10 (bottom-left 2x2 of 4x4)
    # A22_block = A11 (bottom-right 2x2 of 4x4)
    
    # So:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 (top-left 2x2 of C) = M1 - M4 + M5
    # C01 (top-right 2x2 of C) = M3 + M7
    # C10 (bottom-left 2x2 of C) = M2 + M6
    # C11 (bottom-right 2x2 of C) = M1 - M2 + M3
    
    # This is correct.
    
    # Now we build the tensors.
    # Each M is a 2x2x2 Strassen multiplication of 2x2 blocks.
    # So each M contributes 7 rank-1 terms.
    # Total 49 terms.
    
    for m_idx in range(7):
        # Define A, B, C block contributions for this M
        # M1
        if m_idx == 0:
            a_terms = [((0,0), 1), ((1,1), 1)]
            b_terms = [((0,0), 1), ((1,1), 1)]
            c_terms = [((0,0), 1), ((1,1), 1)] # C00 += M1, C11 += M1
        elif m_idx == 1:
            a_terms = [((1,0), 1), ((1,1), 1)]
            b_terms = [((0,0), 1)]
            c_terms = [((1,0), 1), ((1,1), -1)] # C10 += M2, C11 -= M2
        elif m_idx == 2:
            a_terms = [((0,0), 1)]
            b_terms = [((0,1), 1), ((1,1), -1)]
            c_terms = [((0,1), 1), ((1,1), 1)] # C01 += M3, C11 += M3
        elif m_idx == 3:
            a_terms = [((1,1), 1)]
            b_terms = [((1,0), 1), ((0,0), -1)]
            c_terms = [((0,0), -1), ((1,0), 1)] # C00 -= M4, C10 += M4
        elif m_idx == 4:
            a_terms = [((0,0), 1), ((0,1), 1)]
            b_terms = [((1,1), 1)]
            c_terms = [((0,0), 1), ((0,1), -1)] # C00 += M5, C01 -= M5
        elif m_idx == 5:
            a_terms = [((1,0), 1), ((0,0), -1)]
            b_terms = [((0,0), 1), ((0,1), 1)]
            c_terms = [((1,0), 1)] # C10 += M6
        elif m_idx == 6:
            a_terms = [((0,1), 1), ((1,1), -1)]
            b_terms = [((1,0), 1), ((1,1), 1)]
            c_terms = [((0,1), 1)] # C01 += M7
            
        for r2 in range(7):
            col = m_idx * 7 + r2
            
            for (bi, bj), sign_a in a_terms:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            for (bk, bj), sign_b in b_terms:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            for (bi, bj), sign_c in c_terms:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val

    return U, V, W

def _build_4x4x9():
    n, m, p = 4, 4, 9
    rank = 114
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((36, rank), dtype=np.int64)
    W = np.zeros((36, rank), dtype=np.int64)
    
    U444, V444, W444 = _build_strassen_4x4x4()
    
    # Block 1: k in 0..3
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W444[k*4 + i, :]
            
    # Block 2: k in 4..7
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), offset2:offset2+49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, offset2:offset2+49] = W444[k*4 + i, :]
            
    # Block 3: k in 8
    offset3 = 98
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[i*4 + j, offset3 + idx] = 1
            V[j*9 + 8, offset3 + idx] = 1
            W[8*4 + i, offset3 + idx] = 1
            
    return U, V, W

def _validate_numpy(U, V, W, n, m, p):
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_4x4x9()
    
    if not _validate_numpy(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank), dtype=np.int64)
        V_fb = np.zeros((m * p, rank), dtype=np.int64)
        W_fb = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
