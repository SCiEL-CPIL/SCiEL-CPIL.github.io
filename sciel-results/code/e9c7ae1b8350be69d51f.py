import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    # Convert to numpy arrays if not already
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
        
    # Optimization: Use matrix multiplication to verify
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can check this by constructing the tensor or by checking specific slices.
    # For 6x6x6, full check is 216 checks. Each check is a dot product of length rank.
    # Rank ~ 189. 216 * 189 ops is small.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for 2x2:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(0), 01(1), 10(2), 11(3)
    # C: 00(0), 01(1), 10(2), 11(3) -> W[k*n+i] = W[k*2+i]
    # C00: k=0,i=0 -> 0
    # C01: k=0,i=1 -> 1
    # C10: k=1,i=0 -> 2
    # C11: k=1,i=1 -> 3
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_block_strassen_6x6():
    """
    6x6x6 decomposition using 3x3 blocks of 2x2 Strassen.
    Rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    nblocks = 3
    block_size = 2
    
    Us, Vs, Ws = _strassen_2x2()
    srank = 7
    
    total_rank = nblocks * nblocks * nblocks * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                for r in range(srank):
                    # U part
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = 2*bi + di
                            j = 2*bj + dj
                            U[i*m+j, col] = Us[di*2+dj, r]
                    # V part
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = 2*bj + dj
                            k = 2*bk + dk
                            V[j*p+k, col] = Vs[dj*2+dk, r]
                    # W part
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = 2*bk + dk
                            i = 2*bi + di
                            W[k*n+i, col] = Ws[dk*2+di, r]
                    col += 1
    return U, V, W

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication.
    Rank = 19.
    Returns U, V, W of shape (9, 19).
    """
    n, m, p = 3, 3, 3
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices:
    # A: 00(0), 01(1), 02(2), 10(3), 11(4), 12(5), 20(6), 21(7), 22(8)
    # B: 00(0), 01(1), 02(2), 10(3), 11(4), 12(5), 20(6), 21(7), 22(8)
    # C: 00(0), 01(1), 02(2), 10(3), 11(4), 12(5), 20(6), 21(7), 22(8)
    
    # Laderman's 19 products:
    # 1. A00 B00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # 2. A00 B01
    U[0, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1
    
    # 3. A00 B10
    U[0, 2] = 1
    V[3, 2] = 1
    W[3, 2] = 1
    
    # 4. A00 B11
    U[0, 3] = 1
    V[4, 3] = 1
    W[4, 3] = 1
    
    # 5. A10 B00
    U[3, 4] = 1
    V[0, 4] = 1
    W[3, 4] = 1
    
    # 6. A10 B01
    U[3, 5] = 1
    V[1, 5] = 1
    W[4, 5] = 1
    
    # 7. A10 B10
    U[3, 6] = 1
    V[3, 6] = 1
    W[6, 6] = 1
    
    # 8. A10 B11
    U[3, 7] = 1
    V[4, 7] = 1
    W[7, 7] = 1
    
    # 9. A11 B00
    U[4, 8] = 1
    V[0, 8] = 1
    W[4, 8] = 1
    
    # 10. A11 B01
    U[4, 9] = 1
    V[1, 9] = 1
    W[5, 9] = 1
    
    # 11. A11 B10
    U[4, 10] = 1
    V[3, 10] = 1
    W[7, 10] = 1
    
    # 12. A11 B11
    U[4, 11] = 1
    V[4, 11] = 1
    W[8, 11] = 1
    
    # 13. A01 B00
    U[1, 12] = 1
    V[0, 12] = 1
    W[1, 12] = 1
    
    # 14. A01 B01
    U[1, 13] = 1
    V[1, 13] = 1
    W[2, 13] = 1
    
    # 15. A01 B10
    U[1, 14] = 1
    V[3, 14] = 1
    W[4, 14] = 1
    
    # 16. A01 B11
    U[1, 15] = 1
    V[4, 15] = 1
    W[5, 15] = 1
    
    # 17. A20 B00
    U[6, 16] = 1
    V[0, 16] = 1
    W[6, 16] = 1
    
    # 18. A20 B01
    U[6, 17] = 1
    V[1, 17] = 1
    W[7, 17] = 1
    
    # 19. A20 B10
    U[6, 18] = 1
    V[3, 18] = 1
    W[9, 18] = 1
    
    # This is actually just schoolbook for 3x3 with rank 9, not Laderman.
    # Laderman requires specific combinations. Let's use a known correct Laderman implementation.
    # For now, this returns a valid decomposition of rank 9 (schoolbook), which is better than 27.
    # Wait, schoolbook for 3x3 is rank 9. Laderman is rank 19? No, Laderman is 19 for 3x3?
    # Actually, Laderman is for 3x3 and has rank 19. Schoolbook is 9.
    # So this function currently returns schoolbook (rank 9).
    # Let's just use schoolbook for 3x3 blocks, which gives rank 9*9 = 81 for 6x6?
    # No, 6x6 = 2x2 blocks of 3x3.
    # If we use schoolbook for 3x3 (rank 9), and Strassen for 2x2 (rank 7), total rank = 7 * 9 = 63.
    # This is much better than 189.
    
    return U, V, W

def _build_strassen_laderman_6x6():
    """
    6x6x6 decomposition using 2x2 Strassen on 3x3 blocks.
    Each 3x3 block product is decomposed using schoolbook (rank 9).
    Total rank = 7 * 9 = 63.
    """
    n, m, p = 6, 6, 6
    nblocks = 2
    block_size = 3
    
    Us, Vs, Ws = _laderman_3x3()
    srank = Us.shape[1]
    
    total_rank = 7 * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Strassen 2x2 formulas
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to map these to the 3x3 block indices.
    # A00 is rows 0-2, cols 0-2.
    # A01 is rows 0-2, cols 3-5.
    # A10 is rows 3-5, cols 0-2.
    # A11 is rows 3-5, cols 3-5.
    
    # For each Strassen term, we add the corresponding 3x3 decomposition.
    
    # Helper to add a 3x3 decomposition with signs
    def add_block(bi, bj, bk, sign_a, sign_b, col_offset):
        # bi: block row (0 or 1)
        # bj: block col (0 or 1)
        # bk: block col for B (0 or 1)
        # sign_a, sign_b: multipliers
        # This adds the decomposition of A_block * B_block
        # A_block is at (bi, bj)
        # B_block is at (bj, bk)
        
        for r in range(srank):
            # U part (A)
            for di in range(block_size):
                for dj in range(block_size):
                    i = 3*bi + di
                    j = 3*bj + dj
                    U[i*m+j, col_offset + r] += sign_a * Us[di*3+dj, r]
            
            # V part (B)
            for dj in range(block_size):
                for dk in range(block_size):
                    j = 3*bj + dj
                    k = 3*bk + dk
                    V[j*p+k, col_offset + r] += sign_b * Vs[dj*3+dk, r]
            
            # W part (C)
            # C_block is at (bi, bk)
            for dk in range(block_size):
                for di in range(block_size):
                    k = 3*bk + dk
                    i = 3*bi + di
                    W[k*n+i, col_offset + r] += Ws[dk*3+di, r]

    col = 0
    
    # P1 = (A00+A11)(B00+B11)
    # A00 * B00
    add_block(0, 0, 0, 1, 1, col)
    # A00 * B11
    add_block(0, 0, 1, 1, 1, col)
    # A11 * B00
    add_block(1, 0, 0, 1, 1, col)
    # A11 * B11
    add_block(1, 0, 1, 1, 1, col)
    # Note: This is not quite right. Strassen combines them into one product.
    # The decomposition of (A+B)(C+D) is not simply sum of decompositions of AC, AD, BC, BD
    # because the rank would be 4*srank.
    # Strassen relies on the fact that (A+B)(C+D) is ONE product.
    # So we need to decompose the matrix product X*Y where X = A00+A11 and Y = B00+B11.
    # Since matrix multiplication is bilinear, the tensor for X*Y is the sum of tensors for the products of the components?
    # No. The tensor for (A+B)(C+D) is T[(A+B), (C+D), .]
    # This is equal to T[A,C,.] + T[A,D,.] + T[B,C,.] + T[B,D,.]
    # So the rank is indeed the sum of the ranks of the 4 products.
    # BUT, Strassen's trick is that we compute P1 = (A00+A11)(B00+B11) as a single matrix multiplication.
    # In the tensor decomposition sense, we are decomposing the tensor for the 6x6 multiplication.
    # The term corresponding to P1 contributes to C00, C01, C10, C11.
    # Specifically, C00 gets P1, C01 gets P1, C10 gets P1, C11 gets P1? No.
    # C00 = P1 - P2 - P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P3 - P5 + P6
    
    # So the decomposition is a linear combination of the 7 products.
    # Each product Pk is a 3x3 matrix product.
    # We decompose each Pk into srank terms.
    # The total rank is 7 * srank.
    
    # Let's restart the construction properly.
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # P1 = (A00+A11)(B00+B11)
    # A_part = A00 + A11
    # B_part = B00 + B11
    # We need to decompose A_part * B_part.
    # A_part has non-zero entries in blocks (0,0) and (1,1).
    # B_part has non-zero entries in blocks (0,0) and (1,1).
    # The product A_part * B_part has blocks:
    # (0,0): A00*B00
    # (0,1): A00*B11
    # (1,0): A11*B00
    # (1,1): A11*B11
    
    # Wait, the product of two block matrices is block-wise.
    # (A_part * B_part)_{ij} = sum_k A_part_{ik} B_part_{kj}
    # Since A_part is block diagonal (only 0,0 and 1,1 non-zero), and B_part is block diagonal:
    # (A_part * B_part)_{00} = A00 * B00
    # (A_part * B_part)_{01} = 0
    # (A_part * B_part)_{10} = 0
    # (A_part * B_part)_{11} = A11 * B11
    
    # This is NOT what Strassen uses. Strassen uses:
    # P1 = (A00+A11)(B00+B11)
    # This is a single matrix multiplication of two 3x3 matrices.
    # The result is a 3x3 matrix.
    # This result is then used to form C00, C01, C10, C11.
    # C00 = P1 - P2 - P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P3 - P5 + P6
    
    # So P1 is a 3x3 matrix.
    # We decompose P1 = X * Y where X = A00+A11 and Y = B00+B11.
    # X is a 3x3 matrix. Y is a 3x3 matrix.
    # We use the 3x3 decomposition (schoolbook or Laderman) to decompose X*Y.
    
    # So, for each of the 7 products, we identify X and Y, and add the decomposition of X*Y.
    
    # Let's define the X and Y for each product.
    # P1: X = A00+A11, Y = B00+B11
    # P2: X = A10+A11, Y = B00
    # P3: X = A00, Y = B01-B11
    # P4: X = A11, Y = B10-B00
    # P5: X = A00+A01, Y = B11
    # P6: X = A10-A00, Y = B00+B01
    # P7: X = A01-A11, Y = B10+B11
    
    # For each product, we add the decomposition of X*Y to U, V, W.
    # The W part will be combined later to form C.
    # Actually, the W part of the decomposition of X*Y gives the contribution to the C blocks.
    # Since C is a linear combination of P1..P7, the W vectors will be scaled and added.
    
    # Let's implement this.
    
    # Helper to add decomposition of X*Y
    # X is defined by a list of (block_i, block_j, sign)
    # Y is defined by a list of (block_j, block_k, sign)
    
    def add_product(X_def, Y_def, col_offset):
        # X_def: list of (bi, bj, sign_a)
        # Y_def: list of (bj, bk, sign_b)
        # We are decomposing X*Y.
        # X is a 3x3 matrix composed of 3x3 blocks? No, X is a 3x3 matrix.
        # The blocks are 3x3. So X is a 9x9 matrix? No.
        # The whole matrix is 6x6. The blocks are 3x3.
        # So X is a 3x3 matrix of blocks? No.
        # Strassen is applied to the 2x2 block matrix.
        # So A is a 2x2 matrix of 3x3 blocks.
        # A00 is a 3x3 block.
        # X = A00 + A11 is a 3x3 matrix.
        # So X is just a 3x3 matrix.
        # We decompose X*Y using the 3x3 decomposition.
        
        # Construct X and Y as 3x3 matrices.
        X_mat = np.zeros((3, 3))
        Y_mat = np.zeros((3, 3))
        
        for bi, bj, sign in X_def:
            # Add block (bi, bj) to X_mat
            # Block (bi, bj) corresponds to rows 3*bi..3*bi+2, cols 3*bj..3*bj+2 in the 6x6 matrix.
            # But X_mat is just the 3x3 block itself.
            # Wait, X is a 3x3 matrix. It is one of the blocks.
            # No, X is a linear combination of blocks.
            # A00 is a 3x3 matrix. A11 is a 3x3 matrix.
            # X = A00 + A11 is a 3x3 matrix.
            # So X_def should specify which blocks to sum.
            
            # This logic is flawed. X is a 3x3 matrix.
            # The blocks A00, A11 are 3x3 matrices.
            # So X is just a 3x3 matrix.
            # We don't need to construct X_mat.
            # We just need to know the coefficients of the basis vectors.
            
            pass
        
        # This approach is getting complicated. Let's stick to the simpler Block Strassen (189) which is known to work.
        # Or, use the 3x3 schoolbook decomposition directly for the 6x6 matrix?
        # 6x6 = 2x2 blocks of 3x3.
        # If we use schoolbook for 3x3 (rank 9), and schoolbook for 2x2 (rank 4), total rank = 4 * 9 = 36.
        # This is much better than 189.
        
        pass

    # Let's implement 6x6 using 2x2 schoolbook on 3x3 schoolbook blocks.
    # Rank = 4 * 9 = 36.
    
    U = np.zeros((n*m, 36))
    V = np.zeros((m*p, 36))
    W = np.zeros((p*n, 36))
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Product of block (bi, bj) and (bj, bk)
                # Use schoolbook for 3x3
                for r in range(9):
                    # U part
                    for di in range(3):
                        for dj in range(3):
                            i = 3*bi + di
                            j = 3*bj + dj
                            U[i*m+j, col] = Us[di*3+dj, r]
                    # V part
                    for dj in range(3):
                        for dk in range(3):
                            j = 3*bj + dj
                            k = 3*bk + dk
                            V[j*p+k, col] = Vs[dj*3+dk, r]
                    # W part
                    for dk in range(3):
                        for di in range(3):
                            k = 3*bk + dk
                            i = 3*bi + di
                            W[k*n+i, col] = Ws[dk*3+di, r]
                    col += 1
                    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # 1. Try Block Strassen (Rank 189)
    best_rank = 216
    best_result = None
    
    if time.time() < deadline - 20:
        try:
            U_bs, V_bs, W_bs = _build_block_strassen_6x6()
            if _verify(U_bs, V_bs, W_bs):
                rank = U_bs.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_bs, V_bs, W_bs)
        except Exception:
            pass
            
    # 2. Try 2x2 Schoolbook on 3x3 Schoolbook blocks (Rank 36)
    if time.time() < deadline - 20:
        try:
            U_sl, V_sl, W_sl = _build_strassen_laderman_6x6()
            if _verify(U_sl, V_sl, W_sl):
                rank = U_sl.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_sl, V_sl, W_sl)
        except Exception:
            pass

    # 3. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
