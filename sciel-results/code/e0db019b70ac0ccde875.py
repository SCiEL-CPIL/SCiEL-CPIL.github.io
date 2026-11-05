"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based rank reduction for 6x6x6 matrix multiplication.
    
    We partition the 6x6 matrices into 2x2 blocks of 3x3 sub-matrices.
    Instead of doing 8 independent 3x3 multiplications (schoolbook at block level),
    we use Strassen's algorithm which requires only 7 multiplications of 3x3 matrices.
    
    However, Strassen involves additions/subtractions of blocks BEFORE and AFTER 
    the multiplications. In tensor decomposition terms:
    
    For each Strassen multiplication M_k = (linear combo of A blocks) * (linear combo of B blocks),
    we need to expand it into scalar multiplications.
    
    Let's define the 7 Strassen products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    Each M_k is a product of two 3x3 matrices (after linear combinations).
    
    In the tensor decomposition:
    U has shape (36, rank) - indices (i,j) where i in [0,3), j in [0,3) for A
    V has shape (36, rank) - indices (j,k) where j in [0,3), k in [0,3) for B
    W has shape (36, rank) - indices (k,i) where k in [0,3), i in [0,3) for C
    
    Wait, let me re-index properly.
    
    A is 6x6, B is 6x6, C = A*B is 6x6.
    
    Block partition:
    A = [[A00, A01], [A10, A11]] where each block is 3x3
    B = [[B00, B01], [B10, B11]] where each block is 3x3
    C = [[C00, C01], [C10, C11]] where each block is 3x3
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M7
    
    Each M_k involves products of 3x3 matrices. For example:
    M1 = (A00 + A11)(B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    
    So M1 expands to 4 scalar-level 3x3 multiplications.
    Similarly, we need to count the total number of 3x3 scalar multiplications.
    
    Let me count carefully:
    M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11  -> 4 terms
    M2 = (A10+A11)B00 = A10*B00 + A11*B00  -> 2 terms
    M3 = A00(B01-B11) = A00*B01 - A00*B11  -> 2 terms
    M4 = A11(B10-B00) = A11*B10 - A11*B00  -> 2 terms
    M5 = (A00+A01)B11 = A00*B11 + A01*B11  -> 2 terms
    M6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01  -> 4 terms
    M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11  -> 4 terms
    
    Total: 4+2+2+2+2+4+4 = 20 terms of 3x3 multiplication
    
    But wait - some of these terms might cancel in the final C. Let me check:
    
    C00 = M1 - M4 + M5 + M7
    = (A00*B00 + A00*B11 + A11*B00 + A11*B11) 
    - (A11*B10 - A11*B00) 
    + (A00*B11 + A01*B11) 
    + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B10 + A11*B00 + A00*B11 + A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    = A00*B00 + 2*A00*B11 + 2*A11*B00 + A11*B11 - 2*A11*B10 + 2*A01*B11 + A01*B10
    
    Hmm, that doesn't look right. The standard Strassen formulas should give:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Let me recompute. I think I'm making an error in the expansion.
    
    Actually, the key insight is: we don't need to expand into individual scalar multiplications.
    We can keep the Strassen structure and just have 7 "block multiplications" where each block 
    multiplication is itself a 3x3 multiplication.
    
    But in tensor decomposition, each 3x3 multiplication requires rank 27 (schoolbook).
    So 7 * 27 = 189 total rank.
    
    But we also need to handle the linear combinations (additions/subtractions) of the input blocks.
    In tensor decomposition, these are free - they're just linear combinations of the U and V entries.
    
    So the approach is:
    1. For each of the 7 Strassen products, define the linear combination of A-blocks and B-blocks.
    2. Each product is a 3x3 matrix multiplication, which has rank 27.
    3. Total rank = 7 * 27 = 189.
    
    Let me implement this properly.
    
    For a 3x3 matrix multiplication X*Y = Z, where X, Y, Z are 3x3:
    The schoolbook decomposition has 27 terms, one for each (i,j,k) in {0,1,2}^3.
    U_{(xi, xj), c} = 1 where c corresponds to (i,j,k)
    V_{(yj, yk), c} = 1
    W_{(zk, zi), c} = 1
    
    Now for Strassen:
    Let's define the block indices:
    A block (br, bc) where br in {0,1}, bc in {0,1}, each block is 3x3.
    So A[br*3 : (br+1)*3, bc*3 : (bc+1)*3] is the block.
    
    Similarly for B and C.
    
    For each Strassen product M_k, we have:
    M_k = (sum_{blocks} alpha_{k,br,bc} * A_{br,bc}) * (sum_{blocks} beta_{k,br,bc} * B_{br,bc})
    
    This expands to:
    M_k = sum_{br1,bc1,br2,bc2} alpha_{k,br1,bc1} * beta_{k,br2,bc2} * (A_{br1,bc1} * B_{br2,bc2})
    
    Each A_{br1,bc1} * B_{br2,bc2} is a 3x3 * 3x3 = 3x3 multiplication.
    
    So for each (br1,bc1,br2,bc2) with non-zero alpha*beta, we need a full 3x3 schoolbook multiplication.
    
    Let me enumerate the Strassen products and their coefficients:
    
    M1 = (A00 + A11)(B00 + B11)
    alpha: A00:1, A11:1
    beta: B00:1, B11:1
    Products: A00*B00, A00*B11, A11*B00, A11*B11 (4 terms)
    
    M2 = (A10 + A11)B00
    alpha: A10:1, A11:1
    beta: B00:1
    Products: A10*B00, A11*B00 (2 terms)
    
    M3 = A00(B01 - B11)
    alpha: A00:1
    beta: B01:1, B11:-1
    Products: A00*B01, A00*B11 (2 terms, with signs)
    
    M4 = A11(B10 - B00)
    alpha: A11:1
    beta: B10:1, B00:-1
    Products: A11*B10, A11*B00 (2 terms, with signs)
    
    M5 = (A00 + A01)B11
    alpha: A00:1, A01:1
    beta: B11:1
    Products: A00*B11, A01*B11 (2 terms)
    
    M6 = (A10 - A00)(B00 + B01)
    alpha: A10:1, A00:-1
    beta: B00:1, B01:1
    Products: A10*B00, A10*B01, A00*B00, A00*B01 (4 terms, with signs)
    
    M7 = (A01 - A11)(B10 + B11)
    alpha: A01:1, A11:-1
    beta: B10:1, B11:1
    Products: A01*B10, A01*B11, A11*B10, A11*B11 (4 terms, with signs)
    
    Total unique (A_block, B_block) pairs with non-zero coefficients:
    Let me list them all:
    (A00,B00): M1(+1), M6(-1)
    (A00,B11): M1(+1), M3(-1), M5(+1)
    (A11,B00): M1(+1), M2(+1), M4(-1)
    (A11,B11): M1(+1), M7(-1)
    (A10,B00): M2(+1), M6(+1)
    (A10,B01): M6(+1)
    (A00,B01): M3(+1), M6(-1)
    (A11,B10): M4(+1), M7(-1)
    (A01,B11): M5(+1), M7(+1)
    (A01,B10): M7(+1)
    
    So we have 10 distinct (A_block, B_block) pairs, each requiring a 3x3 multiplication (rank 27).
    Total rank = 10 * 27 = 270.
    
    Hmm, that's not 189. The issue is that the same (A_block, B_block) pair appears in multiple Strassen products, so we can't just have 7 separate 3x3 multiplications. We need to account for the fact that the same block product contributes to multiple M_k.
    
    Actually, I think I was overcomplicating this. Let me reconsider.
    
    The standard way to do Strassen in tensor decomposition:
    
    For each of the 7 Strassen products, we define:
    X_k = sum of alpha * A_blocks (a 3x3 matrix)
    Y_k = sum of beta * B_blocks (a 3x3 matrix)
    M_k = X_k * Y_k (a 3x3 matrix)
    
    Then C is assembled from the M_k's.
    
    In tensor decomposition:
    - For each k in 1..7, and for each (i,j) in 3x3 (position in the 3x3 block),
      and for each (j',k') in 3x3:
      We need a term that contributes to C.
    
    But X_k and Y_k are themselves linear combinations of the original A and B blocks.
    
    So for a fixed k, X_k[i,j] = sum_{br,bc} alpha_{k,br,bc} * A[br*3+i, bc*3+j]
    Y_k[j',k'] = sum_{br,bc} beta_{k,br,bc} * B[br*3+j', bc*3+k']
    
    M_k[i,k'] = sum_{j} X_k[i,j] * Y_k[j,k']
    = sum_{j} (sum_{br1,bc1} alpha_{k,br1,bc1} * A[br1*3+i, bc1*3+j]) * (sum_{br2,bc2} beta_{k,br2,bc2} * B[br2*3+j, bc2*3+k'])
    = sum_{br1,bc1,br2,bc2} alpha_{k,br1,bc1} * beta_{k,br2,bc2} * (sum_{j} A[br1*3+i, bc1*3+j] * B[br2*3+j, bc2*3+k'])
    
    Each inner sum is a 3x3 dot product, which in tensor decomposition requires 3 scalar multiplications.
    
    So for each k, and for each (i,k') in 3x3, we have:
    sum over j of X_k[i,j]*Y_k[j,k']
    
    In tensor decomposition, each multiplication A[...]*B[...] is a separate rank-1 term.
    
    Total rank = 7 (Strassen products) * 9 (output positions per 3x3) * 3 (sum over j) = 189.
    
    But wait, some of these terms might be the same scalar multiplication appearing multiple times.
    No, they're distinct because they have different (k, i, j, k') indices.
    
    Actually, no. The scalar multiplication A[a,b]*B[b,c] is the same regardless of which Strassen product it's used in. But in tensor decomposition, each occurrence is a separate rank-1 term because the U, V, W coefficients differ.
    
    Let me just implement it directly:
    
    For each k in range(7):
      For each i in range(3):  # row in 3x3 block
        For each j in range(3):  # column in A block / row in B block
          For each k_prime in range(3):  # column in 3x3 block
    
    Hmm, I think the cleanest way is:
    
    rank = 0
    U = []
    V = []
    W = []
    
    for k in range(7):
      # Define alpha and beta as 2x2 matrices (for the 4 blocks)
      # alpha[br][bc] and beta[br][bc]
      
      # For each (i, j, k_prime) in 3x3x3:
      for i in range(3):
        for j in range(3):
          for kp in range(3):
            # This term contributes to M_k[i, kp] via A[?, j] * B[j, ?]
            # Actually, let me think differently.
            
            # X_k[i, j] = sum_{br,bc} alpha[br][bc] * A[br*3+i, bc*3+j]
            # Y_k[j, kp] = sum_{br,bc} beta[br][bc] * B[br*3+j, bc*3+kp]
            # M_k[i, kp] += X_k[i, j] * Y_k[j, kp]
            
            # So for this (i, j, kp), we need:
            # U part: coefficients for A[br*3+i, bc*3+j] for all (br,bc)
            # V part: coefficients for B[br*3+j, bc*3+kp] for all (br,bc)
            # W part: coefficient for C[?, ?]
            
            # The output position: C block (cr, cc) at position (i, kp)
            # C[cr*3+i, cc*3+kp] += (contribution from M_k)
            
            # M_k contributes to C as:
            # C00 += M1 - M4 + M5 + M7
            # C01 += M3 + M5
            # C10 += M2 + M4
            # C11 += M1 - M3 + M6 + M7
            
            # So for Strassen product k, it contributes to certain C blocks with certain signs.
            
    This is getting complex. Let me just code it up with the full expansion.
    
    """
    import time
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    # Fallback to schoolbook if we run out of time
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def check_time():
        return (time.time() - start_time) > (deadline - safety_margin)
    
    if check_time():
        return best_result
    
    # Strassen decomposition
    # Blocks are 3x3
    bs = 3
    
    # Define Strassen coefficients
    # For each product k (0-indexed), alpha and beta are 2x2 arrays
    # alpha[br][bc] for A blocks, beta[br][bc] for B blocks
    
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_products = [
        # (alpha, beta) where alpha and beta are 2x2 lists
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),  # M0
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),  # M1
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]),  # M2
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]]),  # M3
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),  # M4
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]]),  # M5
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]]),  # M6
    ]
    
    # C assembly coefficients
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M5 + M6
    
    # c_contrib[cr][cc][k] = coefficient of M_k in C_{cr,cc}
    c_contrib = [
        [[1, 0, 0, -1, 1, 0, 1],   # C00
         [0, 0, 1, 0, 1, 0, 0]],   # C01
        [[0, 1, 0, 1, 0, 0, 0],   # C10
         [1, 0, -1, 0, 0, 1, 1]]   # C11
    ]
    
    # Build the decomposition
    U = []
    V = []
    W = []
    
    for k in range(7):
        alpha, beta = strassen_products[k]
        
        for i in range(bs):  # row in 3x3 block
            for j in range(bs):  # column in A / row in B
                for kp in range(bs):  # column in 3x3 block
                    
                    if check_time():
                        return best_result
                    
                    # U row: A[br*bs+i, bc*bs+j] for br,bc in 0,1
                    # V row: B[br*bs+j, bc*bs+kp] for br,bc in 0,1
                    # W row: C[cr*bs+i, cc*bs+kp] for cr,cc in 0,1
                    
                    u_row = [0] * (2 * 2 * bs)  # 4 blocks * 3 columns = 12? No.
                    # U has shape (n*m, rank) = (36, rank)
                    # So U[row][col] where row is (i_a, j_a) in [0,6)x[0,6)
                    # For A[br*bs+i, bc*bs+j], the row index is (br*bs+i) * m + (bc*bs+j)
                    
                    u_row = [0] * (n * m)
                    v_row = [0] * (m * p)
                    w_row = [0] * (p * n)
                    
                    for br in range(2):
                        for bc in range(2):
                            a_idx = (br * bs + i) * m + (bc * bs + j)
                            u_row[a_idx] = alpha[br][bc]
                            
                            b_idx = (br * bs + j) * p + (bc * bs + kp)
                            v_row[b_idx] = beta[br][bc]
                    
                    for cr in range(2):
                        for cc in range(2):
                            c_idx = (cr * bs + i) * n + (cc * bs + kp)
                            w_row[c_idx] = c_contrib[cr][cc][k]
                    
                    U.append(u_row)
                    V.append(v_row)
                    W.append(w_row)
    
    rank = len(U)
    
    if check_time():
        return best_result
    
    # Convert to proper format
    U_final = [[0] * rank for _ in range(n * m)]
    V_final = [[0] * rank for _ in range(m * p)]
    W_final = [[0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        for row_idx in range(n * m):
            U_final[row_idx][col_idx] = U[col_idx][row_idx]
        for row_idx in range(m * p):
            V_final[row_idx][col_idx] = V[col_idx][row_idx]
        for row_idx in range(p * n):
            W_final[row_idx][col_idx] = W[col_idx][row_idx]
    
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    return result
# EVOLVE-BLOCK-END
