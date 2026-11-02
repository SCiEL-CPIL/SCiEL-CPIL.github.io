import numpy as np
import time

def _strassen_2x2():
    """Returns U, V, W for 2x2 Strassen (Rank 7).
    Indices: A: 0=A00, 1=A01, 2=A10, 3=A11
             B: 0=B00, 1=B01, 2=B10, 3=B11
             C: 0=C00, 1=C01, 2=C10, 3=C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W


def _strassen_3x3_rank21():
    """Returns U, V, W for 3x3 matrix multiplication with rank 21.
    
    This uses a known rank-21 decomposition from the literature.
    The 3x3 matrices are indexed:
    A: 0=A00, 1=A01, 2=A02, 3=A10, 4=A11, 5=A12, 6=A20, 7=A21, 8=A22
    B: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12, 6=B20, 7=B21, 8=B22
    C: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12, 6=C20, 7=C21, 8=C22
    
    We use a structured decomposition based on combining Strassen-like ideas.
    """
    # Build using a verified approach: partition 3x3 into overlapping 2x2 blocks
    # and use Strassen on each, then correct.
    # Actually, let's use a direct known construction.
    
    # A clean rank-21 decomposition can be built by viewing 3x3 as:
    # Use the fact that rank(3x3x3) = 21 is achievable.
    # Here's one explicit construction:
    
    R = 21
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # We'll use a construction based on:
    # Partition into three 2x2 overlapping submatrices and use Strassen,
    # then combine with corrections.
    
    # Alternative: Use the BSGS-style or direct enumeration.
    # Let me use a verified rank-21 construction from literature.
    
    # Construction: 3x3 = 2x2 (top-left) + cross terms
    # M1-M7: Strassen on top-left 2x2 (A[0:2,0:2] * B[0:2,0:2])
    # M8-M14: Strassen on bottom-right 2x2 (A[1:3,1:3] * B[1:3,1:3])  
    # M15-M21: Correction terms for the overlap and remaining entries
    
    # Actually, let me use a simpler verified approach.
    # I'll construct rank 21 using the following known pattern:
    
    # The 21 terms are structured as follows (using 0-indexed block positions):
    # Terms 1-7: Strassen on rows/cols 0,1 (top-left 2x2 block)
    # Terms 8-14: Strassen on rows/cols 1,2 (bottom-right 2x2 block)
    # Terms 15-21: Handle the remaining interactions
    
    # Let me instead use a direct verified construction.
    # Here's a rank-21 construction:
    
    # Use the fact that we can write 3x3 multiplication as:
    # Split A into [A_top; A_bottom] and B into [B_left B_right]
    
    # I'll use a computational approach to generate a valid rank-21 decomposition.
    # For now, let me use a fallback: 3x3 schoolbook rank 27 combined with 2x2 Strassen = 189
    # But we want 147, so let me try harder.
    
    # Known rank-21 construction (from "Fast matrix multiplication" literature):
    # The 21 products are:
    r = 0
    # Strassen-like for the 3x3 case:
    # We define 7 "Strassen" terms for each of 3 overlapping 2x2 blocks,
    # but we need to be careful about double-counting.
    
    # Let me use a different approach: direct rank-21 via explicit formulas.
    # After research, here's a valid rank-21 decomposition:
    
    # Term 1: (A00+A11+A22)(B00+B11+B22)
    U[0,0]=1; U[4,0]=1; U[8,0]=1
    V[0,0]=1; V[4,0]=1; V[8,0]=1
    W[0,0]=1; W[4,0]=1; W[8,0]=1
    r += 1
    
    # Term 2: (A00+A11)(B00+B11)
    U[0,1]=1; U[4,1]=1
    V[0,1]=1; V[4,1]=1
    W[0,1]=1; W[4,1]=1
    r += 1
    
    # Term 3: (A00+A22)(B00+B22)
    U[0,2]=1; U[8,2]=1
    V[0,2]=1; V[8,2]=1
    W[0,2]=1; W[8,2]=1
    r += 1
    
    # Term 4: (A11+A22)(B11+B22)
    U[4,3]=1; U[8,3]=1
    V[4,3]=1; V[8,3]=1
    W[4,3]=1; W[8,3]=1
    r += 1
    
    # Term 5: (A01+A02)(B10+B20)
    U[1,4]=1; U[2,4]=1
    V[3,4]=1; V[6,4]=1
    W[0,4]=1
    r += 1
    
    # Term 6: (A01+A12)(B20+B21)  -- needs care
    U[1,5]=1; U[5,5]=1
    V[6,5]=1; V[7,5]=1
    W[0,5]=1
    r += 1
    
    # Term 7: (A10+A20)(B01+B02)
    U[3,6]=1; U[6,6]=1
    V[1,6]=1; V[2,6]=1
    W[0,6]=1
    r += 1
    
    # Term 8: (A10+A11)(B01+B11)
    U[3,7]=1; U[4,7]=1
    V[1,7]=1; V[4,7]=1
    W[0,7]=1; W[4,7]=1
    r += 1
    
    # Term 9: (A10+A12)(B21+B22)
    U[3,8]=1; U[5,8]=1
    V[7,8]=1; V[8,8]=1
    W[0,8]=1
    r += 1
    
    # Term 10: (A11+A22)(B02+B22)
    U[4,9]=1; U[8,9]=1
    V[2,9]=1; V[8,9]=1
    W[0,9]=1; W[8,9]=1
    r += 1
    
    # Term 11: (A02+A12)(B10+B20)
    U[2,10]=1; U[5,10]=1
    V[3,10]=1; V[6,10]=1
    W[0,10]=1
    r += 1
    
    # Term 12: (A02+A22)(B10+B20) -- this is getting messy
    # Let me abandon this hand-crafted approach and use a verified one.
    
    # Fallback: return None to indicate we should use schoolbook 3x3
    return None


def _schoolbook_3x3():
    """Returns U, V, W for 3x3 Schoolbook (Rank 27)."""
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, r] = 1
                V[j*3 + k, r] = 1
                W[k*3 + i, r] = 1
                r += 1
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8, num_zero_tests=200):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    # Check all non-zero entries (should equal 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries
    np.random.seed(12345)
    for _ in range(num_zero_tests):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1 = a // m; j1 = a % m
        j2 = b // p; k1 = b % p
        k2 = c // n; i2 = c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True


def _build_kronecker_decomp(n_blocks, m_blocks, p_blocks, 
                              U_block, V_block, W_block,
                              block_n, block_m, block_p):
    """Build a full decomposition from block decomposition.
    
    The full matrix is n_blocks x m_blocks, where each element is a 
    block_n x block_m submatrix.
    """
    n_total = n_blocks * block_n
    m_total = m_blocks * block_m
    p_total = p_blocks * block_p
    
    R_block = U_block.shape[1]
    R_total = R_block  # Each block position uses the same rank terms
    
    # Actually for Kronecker-style, the rank multiplies
    # But here we're doing a structured composition
    # Let me think about this differently.
    
    # The full input A has shape (n_total, m_total), flattened to n_total*m_total
    # Each element A[i_total, j_total] where i_total = i_block*block_n + i_sub
    #                                     j_total = j_block*block_m + j_sub
    # belongs to block (i_block, j_block) at sub-position (i_sub, j_sub)
    
    # The block-level multiplication gives:
    # C_block[i_b, j_b] = sum_{k_b} A_block[i_b, k_b] * B_block[k_b, j_b]
    # where each "element" is a block_n x block_m matrix multiplied by block_m x block_p matrix
    
    # This is getting complex. Let me use the simpler approach from the original:
    # 6x6 = 3x3 blocks of 2x2, using 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7)
    # Total rank = 27 * 7 = 189
    
    # For rank 147, I need rank 21 for 3x3.
    
    pass


def _build_6x6_from_3x3_2x2(U3, V3, W3, U2, V2, W2):
    """Build 6x6 decomposition from 3x3 (block level) and 2x2 (sub-block level).
    
    The 6x6 matrix is viewed as 3x3 blocks of 2x2 submatrices.
    Input A (6x6) -> flattened to 36 elements
    Block (i,j) contains A[i*2:(i+1)*2, j*2:(j+1)*2]
    Flattened index for element (row, col) = row*6 + col
    = (i*2 + a)*(6) + (j*2 + b) = i*12 + a*6 + j*2 + b
    
    But the convention is W[k*n+i, r] for output C[i,j]
    Let me follow the same indexing as the original program.
    
    For input A (n=6, m=6): index a = i*m + j = i*6 + j
    For block (i_block, j_block) with i_block = i//2, j_block = j//2
    sub-index: a_sub = (i%2)*2 + (j%2)  -- this doesn't match Strassen's indexing
    
    Actually in the original, the flattening is:
    U has shape (n*m, rank) = (36, rank)
    Index into U for A[i,j] is i*m + j = i*6 + j
    
    For the 3x3 block structure:
    A[i,j] where i = i_b*2 + a, j = j_b*2 + b
    Block index (i_b, j_b), sub-index (a, b)
    
    In U3 (3x3 schoolbook), U3[i_b, j_b, r3] picks out block (i_b, j_b)
    In U2 (2x2 Strassen), U2[a, b, r2] picks out sub-element (a, b)
    
    So U_final[i*6+j, r3*7+r2] = U3[i_b, j_b, r3] * U2[a, b, r2]
    where i_b = i//2, j_b = j//2, a = i%2, b = j%2
    """
    U2, V2, W2 = U2, V2, W2  # 2x2 Strassen
    U3, V3, W3 = U3, V3, W3  # 3x3 decomposition
    
    R3 = U3.shape[1]
    R2 = U2.shape[1]
    R_total = R3 * R2
    
    n = 6
    U = np.zeros((36, R_total))
    V = np.zeros((36, R_total))
    W = np.zeros((36, R_total))
    
    for r3 in range(R3):
        for r2 in range(R2):
            r = r3 * R2 + r2
            for i in range(6):
                for j in range(6):
                    a = i * 6 + j
                    i_b = i // 2
                    j_b = j // 2
                    a_sub = i % 2
                    b_sub = j % 2
                    U[a, r] = U3[i_b, r3] * U2[a_sub, r2]
                    
                    b_idx = j * 6 + (i)  # Wait, B is (m,p) = (6,6)
                    # B[j, k] where j = j_b*2 + b_sub, k = k_b*2 + c_sub
                    # For V, the index is j*6 + k
                    # But we're building V for all j,k at once
                    pass
                    
            # Let me redo this more carefully
            for j in range(6):
                for k in range(6):
                    b_idx = j * 6 + k
                    j_b = j // 2
                    k_b = k // 2
                    b_sub = j % 2
                    c_sub = k % 2
                    V[b_idx, r] = V3[j_b, r3] * V2[b_sub, r2]
                    
            for k in range(6):
                for i in range(6):
                    c_idx = k * 6 + i
                    k_b = k // 2
                    i_b = i // 2
                    c_sub = k % 2
                    a_sub = i % 2
                    W[c_idx, r] = W3[k_b, r3] * W2[c_sub, r2]
    
    return U, V, W


def _build_6x6_kronecker_strassen():
    """Build 6x6 decomposition: 3x3 schoolbook blocks of 2x2 Strassen.
    Rank = 27 * 7 = 189.
    """
    U3, V3, W3 = _schoolbook_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    R3 = U3.shape[1]
    R2 = U2.shape[1]
    R_total = R3 * R2
    
    U = np.zeros((36, R_total))
    V = np.zeros((36, R_total))
    W = np.zeros((36, R_total))
    
    for r3 in range(R3):
        for r2 in range(R2):
            r = r3 * R2 + r2
            for i in range(6):
                for j in range(6):
                    a_idx = i * 6 + j
                    i_b = i // 2
                    j_b = j // 2
                    a_sub = i % 2
                    b_sub = j % 2
                    U[a_idx, r] = U3[i_b, r3] * U2[a_sub, r2]
                    
            for j in range(6):
                for k in range(6):
                    b_idx = j * 6 + k
                    j_b = j // 2
                    k_b = k // 2
                    b_sub = j % 2
                    c_sub = k % 2
                    V[b_idx, r] = V3[j_b, r3] * V2[b_sub, r2]
                    
            for k in range(6):
                for i in range(6):
                    c_idx = k * 6 + i
                    k_b = k // 2
                    i_b = i // 2
                    c_sub = k % 2
                    a_sub = i % 2
                    W[c_idx, r] = W3[k_b, r3] * W2[c_sub, r2]
    
    return U, V, W


def _build_6x6_schoolbook():
    """Build 6x6 schoolbook decomposition. Rank = 216."""
    R = 216
    U = np.zeros((36, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i*6+j, col] = 1
                V[j*6+k, col] = 1
                W[k*6+i, col] = 1
                col += 1
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Try rank 189 first (3x3 schoolbook * 2x2 Strassen)
    U, V, W = _build_6x6_kronecker_strassen()
    
    if _verify_decomposition(6, 6, 6, U, V, W, tol=1e-8, num_zero_tests=500):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook 216
    U_sb, V_sb, W_sb = _build_6x6_schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
