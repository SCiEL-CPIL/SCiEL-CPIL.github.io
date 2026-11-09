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
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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

def _schoolbook_3x3():
    """
    Schoolbook decomposition for 3x3 matrix multiplication.
    Returns U, V, W of shape (9, 9) with correct trace convention W[k*3+i, r].
    """
    n, m, p = 3, 3, 3
    rank = 9
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

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
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

def _build_2x2_schoolbook_on_3x3():
    """
    6x6x6 decomposition using 2x2 schoolbook on 3x3 schoolbook blocks.
    Rank = 4 * 9 = 36.
    
    The 6x6 matrix is divided into 2x2 blocks of 3x3.
    For each block product (bi,bj)*(bj,bk), we use the 3x3 schoolbook decomposition.
    """
    n, m, p = 6, 6, 6
    nblocks = 2
    block_size = 3
    
    Us, Vs, Ws = _schoolbook_3x3()
    srank = Us.shape[1]  # 9
    
    total_rank = nblocks * nblocks * nblocks * srank  # 8 * 9 = 72? No, 4*9=36
    # Actually: for each (bi, bj, bk) triple, we have one 3x3 product decomposition
    # bi in {0,1}, bj in {0,1}, bk in {0,1}
    # But not all combinations are valid - we need bj to match
    # In schoolbook: C[bi,bk] = sum_bj A[bi,bj] * B[bj,bk]
    # So for each (bi, bk), we sum over bj
    # Total number of (bi, bj, bk) triples = 2*2*2 = 8
    # But each contributes rank 9
    # Total rank = 8 * 9 = 72? No wait...
    
    # Actually in schoolbook for 2x2 blocks:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # That's 8 products of 3x3 matrices, each of rank 9.
    # Total rank = 8 * 9 = 72.
    
    # But wait, that's not right either. Let me think again.
    # The full 6x6 schoolbook has rank 216.
    # If we group into 3x3 blocks, we have 4 block positions in A, 4 in B, 4 in C.
    # The block-level schoolbook has rank 4 (one for each (bi,bj,bk) triple where the blocks overlap).
    # Actually no. The block-level matrix multiplication is:
    # C_blocks = A_blocks * B_blocks where each is 2x2 of 3x3 blocks.
    # C_blocks[bi,bk] = sum_bj A_blocks[bi,bj] * B_blocks[bj,bk]
    # This is standard 2x2 matrix multiplication at the block level.
    # The rank of 2x2 matrix multiplication is 4 (schoolbook).
    # Each "product" at the block level is a 3x3 matrix multiplication of rank 9.
    # So total rank = 4 * 9 = 36? No, that's not how it works.
    
    # Let me be more careful. The tensor for 6x6 matrix multiplication can be decomposed as:
    # For each (i,j,k) in the 6x6 indices, we need a term.
    # i = 3*bi + di, j = 3*bj + dj, k = 3*bk + dk
    # where bi,bj,bk in {0,1} and di,dj,dk in {0,1,2}
    
    # The schoolbook decomposition has one term for each (i,j,k):
    # U[i*6+j, r] = 1, V[j*6+k, r] = 1, W[k*6+i, r] = 1
    
    # If we group by blocks, for fixed (bi,bj,bk), we have 3*3*3 = 27 terms.
    # There are 2*2*2 = 8 block triples.
    # Total = 8 * 27 = 216. Correct.
    
    # Now, within each block triple (bi,bj,bk), the 27 terms form a 3x3x3 tensor decomposition.
    # This is exactly the schoolbook decomposition for 3x3 matrix multiplication, which has rank 27.
    # But we can do better! The 3x3 schoolbook has rank 27, but maybe we can use a lower rank?
    # No, schoolbook for 3x3 is rank 27 (since 3*3*3 = 27).
    
    # Wait, I'm confusing myself. Let me restart.
    
    # The 3x3 schoolbook decomposition:
    # For each (i,j,k) in [0,3)^3, we have one term.
    # U[i*3+j, r] = 1, V[j*3+k, r] = 1, W[k*3+i, r] = 1
    # This has rank 27.
    
    # But earlier I said the 3x3 schoolbook has rank 9. That was wrong!
    # 3x3 schoolbook has rank 3*3*3 = 27, not 9.
    # 2x2 schoolbook has rank 2*2*2 = 8.
    # 2x2 Strassen has rank 7.
    
    # So the 6x6 decomposition using 2x2 schoolbook on 3x3 schoolbook blocks:
    # Rank = 8 * 27 = 216. Same as direct schoolbook!
    
    # That makes sense - it's just a regrouping.
    
    # To get a lower rank, we need to use Strassen at the block level:
    # 2x2 Strassen (rank 7) on 3x3 schoolbook blocks (rank 27 each).
    # But Strassen combines blocks, so it's not simply 7 * 27.
    
    # Actually, the correct way to think about it:
    # The 6x6 matrix multiplication tensor can be seen as a 2x2 block matrix multiplication
    # where each block is 3x3.
    # The rank of this tensor is at most rank(2x2 mult) * rank(3x3 mult) = 8 * 27 = 216 (schoolbook)
    # or 7 * 27 = 189 (Strassen at block level).
    
    # But there might be a better decomposition that doesn't factor through the block structure.
    
    # For now, let's implement the 2x2 Strassen on 3x3 schoolbook blocks (rank 189).
    
    # Actually, let me just implement the direct 6x6 schoolbook (rank 216) and the
    # 2x2 Strassen on 3x3 schoolbook blocks (rank 189).
    
    # For the 2x2 Strassen on 3x3 blocks:
    # We use the Strassen formulas at the block level.
    # Each "product" in Strassen is a 3x3 matrix multiplication.
    # We decompose each 3x3 product using schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
    total_rank = 7 * 27
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Strassen 2x2 formulas (at block level):
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each product, we need to decompose the 3x3 matrix multiplication.
    # The X and Y matrices are 3x3 blocks.
    
    # Helper: add decomposition of X*Y where X is a 3x3 block and Y is a 3x3 block.
    # X is at block position (bi, bj) in A, Y is at block position (bj, bk) in B.
    # The result contributes to block (bi, bk) in C.
    
    def add_3x3_product(bi, bj, bk, sign_a, sign_b, col_offset):
        """Add the schoolbook decomposition of A[bi,bj] * B[bj,bk] to columns starting at col_offset."""
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi + di
                    j = 3*bj + dj
                    k = 3*bk + dk
                    U[i*m+j, col_offset] = sign_a
                    V[j*p+k, col_offset] = sign_b
                    W[k*n+i, col_offset] = 1.0
                    col_offset += 1
        return col_offset
    
    col = 0
    
    # P1 = (A00+A11)(B00+B11)
    # This is a single 3x3 matrix multiplication where X = A00+A11 and Y = B00+B11.
    # But X and Y are sums of blocks, not single blocks.
    # We need to decompose (A00+A11)*(B00+B11).
    # By bilinearity: (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # So we add all four 3x3 products.
    col = add_3x3_product(0, 0, 0, 1, 1, col)   # A00*B00
    col = add_3x3_product(0, 0, 1, 1, 1, col)   # A00*B11
    col = add_3x3_product(1, 0, 0, 1, 1, col)   # A11*B00
    col = add_3x3_product(1, 0, 1, 1, 1, col)   # A11*B11
    
    # P2 = (A10+A11)B00
    col = add_3x3_product(1, 0, 0, 1, 1, col)   # A10*B00
    col = add_3x3_product(1, 1, 0, 1, 1, col)   # A11*B00
    
    # P3 = A00(B01-B11)
    col = add_3x3_product(0, 0, 1, 1, 1, col)   # A00*B01
    col = add_3x3_product(0, 1, 1, 1, -1, col)  # A00*B11 (with -1 on B)
    
    # P4 = A11(B10-B00)
    col = add_3x3_product(1, 1, 0, 1, 1, col)   # A11*B10
    col = add_3x3_product(1, 0, 0, 1, -1, col)  # A11*B00 (with -1 on B)
    
    # P5 = (A00+A01)B11
    col = add_3x3_product(0, 0, 1, 1, 1, col)   # A00*B11
    col = add_3x3_product(0, 1, 1, 1, 1, col)   # A01*B11
    
    # P6 = (A10-A00)(B00+B01)
    col = add_3x3_product(1, 0, 0, 1, 1, col)   # A10*B00
    col = add_3x3_product(1, 0, 1, 1, 1, col)   # A10*B01
    col = add_3x3_product(0, 0, 0, -1, 1, col)  # A00*B00 (with -1 on A)
    col = add_3x3_product(0, 0, 1, -1, 1, col)  # A00*B01 (with -1 on A)
    
    # P7 = (A01-A11)(B10+B11)
    col = add_3x3_product(0, 1, 0, 1, 1, col)   # A01*B10
    col = add_3x3_product(0, 1, 1, 1, 1, col)   # A01*B11
    col = add_3x3_product(1, 1, 0, -1, 1, col)  # A11*B10 (with -1 on A)
    col = add_3x3_product(1, 1, 1, -1, 1, col)  # A11*B11 (with -1 on A)
    
    # Now we need to combine the W parts to form C.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # But wait, I've already added the W parts with coefficient 1 for each product.
    # I need to scale them according to the Strassen formulas.
    
    # Let me redo this more carefully.
    
    # Actually, the issue is that in the tensor decomposition, each "rank-1 term" has
    # a U part, V part, and W part. The W part determines which output element gets the contribution.
    # In Strassen, the output is a linear combination of the products.
    # So the W part of each rank-1 term should encode which output block it contributes to,
    # and with what sign.
    
    # Let me restart with a cleaner approach.
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # For each Strassen product, we add rank-1 terms for each (di,dj,dk) in the 3x3 block.
    # The W part will be set to contribute to the appropriate C block with the appropriate sign.
    
    # P1 = (A00+A11)(B00+B11)
    # Contributes to: C00 (+1), C11 (-1)
    # A part: A00 and A11
    # B part: B00 and B11
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                i0 = 3*0 + di  # A00 row
                j0 = 3*0 + dj  # A00 col / B00 row
                k0 = 3*0 + dk  # B00 col / C00 col
                i1 = 3*1 + di  # A11 row
                j1 = 3*1 + dj  # A11 col / B11 row
                k1 = 3*1 + dk  # B11 col / C11 col
                
                # Term for A00*B00 contributing to C00
                U[i0*m+j0, col] = 1
                V[j0*p+k0, col] = 1
                W[k0*n+i0, col] = 1  # C00
                col += 1
                
                # Term for A00*B11 contributing to C11
                U[i0*m+j1, col] = 1
                V[j1*p+k1, col] = 1
                W[k1*n+i0, col] = -1  # C11
                col += 1
                
                # Term for A11*B00 contributing to C11
                U[i1*m+j0, col] = 1
                V[j0*p+k0, col] = 1
                W[k0*n+i1, col] = -1  # C11
                col += 1
                
                # Term for A11*B11 contributing to C11
                U[i1*m+j1, col] = 1
                V[j1*p+k1, col] = 1
                W[k1*n+i1, col] = -1  # C11
                col += 1
    
    # This is getting very complex. Let me just use the simpler approach:
    # Build the 2x2 Strassen on 3x3 blocks by directly constructing the U, V, W matrices.
    
    # Actually, let me just return the schoolbook for now and verify it works.
    # The key insight is that the current program's rank 36 attempt was wrong because
    # 3x3 schoolbook has rank 27, not 9.
    
    # Let me just implement the 2x2 Strassen on 3x3 schoolbook blocks properly.
    
    # Clear and restart
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Define Strassen products with their A, B, and C contributions.
    # Each product is a 3x3 matrix multiplication.
    # We'll add rank-1 terms for each (di,dj,dk) in the 3x3 block.
    
    # Product definition: (A_blocks, B_blocks, C_contributions)
    # A_blocks: list of (bi, bj, sign)
    # B_blocks: list of (bj, bk, sign)
    # C_contributions: list of (bi, bk, sign)
    
    products = [
        # P1 = (A00+A11)(B00+B11), C00: +1, C11: -1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (0,1,1), (1,0,1), (1,1,1) ], [ (0,0,1), (1,1,-1) ]),
        # P2 = (A10+A11)B00, C10: +1, C11: -1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # P3 = A00(B01-B11), C01: +1, C11: +1
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # P4 = A11(B10-B00), C00: +1, C10: +1
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # P5 = (A00+A01)B11, C00: -1, C01: +1
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # P6 = (A10-A00)(B00+B01), C11: +1
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # P7 = (A01-A11)(B10+B11), C00: +1
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
    ]
    
    for A_blocks, B_blocks, C_contribs in products:
        for (bi_a, bj_a, sign_a) in A_blocks:
            for (bj_b, bk_b, sign_b) in B_blocks:
                if bj_a != bj_b:
                    continue  # Blocks don't connect
                for (bi_c, bk_c, sign_c) in C_contribs:
                    if bi_a != bi_c or bk_b != bk_c:
                        continue  # Output block doesn't match
                    for di in range(3):
                        for dj in range(3):
                            for dk in range(3):
                                i = 3*bi_a + di
                                j = 3*bj_a + dj
                                k = 3*bk_b + dk
                                U[i*m+j, col] = sign_a
                                V[j*p+k, col] = sign_b
                                W[k*n+i, col] = sign_c
                                col += 1
    
    # Check that we used all columns
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = 216
    best_result = None
    
    # 1. Try 2x2 Strassen on 3x3 schoolbook blocks (Rank 189)
    if time.time() < deadline - 20:
        try:
            U_s, V_s, W_s = _build_2x2_schoolbook_on_3x3()
            if _verify(U_s, V_s, W_s):
                rank = U_s.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_s, V_s, W_s)
        except Exception:
            pass
            
    # 2. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
