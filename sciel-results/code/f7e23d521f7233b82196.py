"""Strassen-based rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Blocks: A is 2x4, B is 4x8.
    # A_{11} = rows 0, cols 0-3 (4x4)
    # A_{12} = rows 0, cols 4-7 -- wait, A is 2x4 so it has 2 rows and 4 cols.
    # Let me re-read: "multiplying a 2 by 4 matrix by a 4 by 8 matrix"
    # So A is 2x4, B is 4x8, C is 2x8.
    
    # For Strassen, we treat A as having 2 row-blocks and 2 col-blocks.
    # A has shape (2, 4). If we split rows into 2 blocks of 1, and cols into 2 blocks of 2:
    # A_{11} is (1,2), A_{12} is (1,2), A_{21} is (1,2), A_{22} is (1,2)
    # B has shape (4, 8). Split rows into 2 blocks of 2, cols into 2 blocks of 4:
    # B_{11} is (2,4), B_{12} is (2,4), B_{21} is (2,4), B_{22} is (2,4)
    # C has shape (2, 8). Split rows into 2 blocks of 1, cols into 2 blocks of 4:
    # C_{11} is (1,4), C_{12} is (1,4), C_{21} is (1,4), C_{22} is (1,4)
    
    # Each block multiplication is: (1x2) * (2x4) = (1x4)
    # Each such block product has rank 1*2*4 = 8 in the schoolbook decomposition.
    # Strassen uses 7 block multiplications instead of 4, so 7 * 8 = 56 total rank.
    
    rank = 56
    
    # U has shape (n*m, rank) = (8, 56)
    # V has shape (m*p, rank) = (32, 56)
    # W has shape (p*n, rank) = (16, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 7 Strassen products:
    # M1 = A11 * (B11 + B12)
    # M2 = (A11 + A22) * B11
    # M3 = (A21 - A11) * (B11 + B12)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 + A12) * (B22 - B11)
    # M7 = (A22 - A12) * (B11 + B12)
    
    # Wait, let me use the standard Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Block index mapping:
    # A_{row_block, col_block}: row_block in {0,1}, col_block in {0,1}
    # A_{0,0} = A11, A_{0,1} = A12, A_{1,0} = A21, A_{1,1} = A22
    # B_{row_block, col_block}: row_block in {0,1}, col_block in {0,1}
    # B_{0,0} = B11, B_{0,1} = B12, B_{1,0} = B21, B_{1,1} = B22
    # C_{row_block, col_block}: row_block in {0,1}, col_block in {0,1}
    # C_{0,0} = C11, C_{0,1} = C12, C_{1,0} = C21, C_{1,1} = C22
    
    # For each block, the indices are:
    # A_{rb, cb}: rows rb*1 to rb*1+0 (1 row), cols cb*2 to cb*2+1 (2 cols)
    # So A_{rb, cb} has 1*2 = 2 elements.
    # B_{rb, cb}: rows rb*2 to rb*2+1 (2 rows), cols cb*4 to cb*4+3 (4 cols)
    # So B_{rb, cb} has 2*4 = 8 elements.
    # C_{rb, cb}: rows rb*1 to rb*1+0 (1 row), cols cb*4 to cb*4+3 (4 cols)
    # So C_{rb, cb} has 1*4 = 4 elements.
    
    # Each block product (1x2)*(2x4) = (1x4) has 1*2*4 = 8 rank-1 terms.
    # For block product A_{rb,cb} * B_{rb',cb'} = C_{rb'',cb''}:
    # For each a_row in [rb*1, rb*1+1), a_col in [cb*2, cb*2+2), b_row in [rb'*2, rb'*2+2), b_col in [cb''*4, cb''*4+4):
    #   U[a_row*m + a_col] += coeff
    #   V[b_row*p + b_col] += coeff  (wait, no, the coefficient is part of the rank-1 term)
    
    # Actually, let me think about this more carefully.
    # The tensor decomposition: for each rank-1 term, we have U[i,j], V[j,k], W[k,i] (where i indexes A, j indexes B, k indexes C in a specific way).
    
    # In the schoolbook: for each (i, j, k) where i is row of A, j is col of A / row of B, k is col of B:
    #   U[i*m + j] = 1
    #   V[j*p + k] = 1
    #   W[k*n + i] = 1
    
    # For a block product, we need to decompose the bilinear form.
    # Block A_{rb,cb} has entries A[i, j] where i = rb (only one row since block height is 1), j ranges over 2 cols.
    # Block B_{rb',cb'} has entries B[j, k] where j ranges over 2 rows, k ranges over 4 cols.
    # Block C_{rb'',cb''} has entries C[i, k] where i = rb'' (only one row), k ranges over 4 cols.
    
    # The product C_{rb'',cb''} = A_{rb,cb} * B_{rb',cb'} means:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # where i = rb (single value), j in [cb*2, cb*2+2), k in [cb''*4, cb''*4+4)
    # And B's row index j corresponds to: j_global = rb'*2 + j_local
    
    # So for the block product, the rank-1 terms are indexed by (j_local) and (k_local):
    # For each j_local in [0,2), k_local in [0,4):
    #   i = rb, j_global = cb*2 + j_local, k_global = cb''*4 + k_local
    #   U[i*m + j_global] += coeff
    #   V[j_global*p + k_global] += coeff
    #   W[k_global*n + i] += coeff
    
    # Wait, but B's row index in the block is rb'*2 + j_local, so j_global for V should be rb'*2 + j_local.
    
    # Let me redefine:
    # A_{rb, cb}: A[i, j] where i = rb, j = cb*2 + j_local, j_local in [0,2)
    # B_{rb', cb'}: B[j, k] where j = rb'*2 + j_local, k = cb'*4 + k_local, j_local in [0,2), k_local in [0,4)
    # C_{rb'', cb''}: C[i, k] where i = rb'', k = cb''*4 + k_local, k_local in [0,4)
    
    # For block product A_{rb,cb} * B_{rb',cb'} = C_{rb'',cb''}:
    # C[rb'', k] = sum_{j_local} A[rb, cb*2+j_local] * B[rb'*2+j_local, cb'*4+k_local]
    
    # The rank-1 decomposition for this block product:
    # For each (j_local, k_local):
    #   U[rb*m + (cb*2 + j_local)] += coeff
    #   V[(rb'*2 + j_local)*p + (cb'*4 + k_local)] += coeff
    #   W[(cb''*4 + k_local)*n + rb''] += coeff
    
    # Now, for each Strassen product M_s, we have a linear combination of block products.
    # Each block product contributes with a coefficient (+1 or -1).
    
    # M1 = (A11 + A22)(B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each of these 4 block products contributes with coeff +1.
    
    # For each such contribution (A_{rb,cb} * B_{rb',cb'} -> C_{rb'',cb''}) with coeff c:
    #   For each (j_local, k_local):
    #     U[rb*m + (cb*2 + j_local)] += c
    #     V[(rb'*2 + j_local)*p + (cb'*4 + k_local)] += c
    #     W[(cb''*4 + k_local)*n + rb''] += c
    
    # Let me define the Strassen products as lists of (A_rb, A_cb, B_rb, B_cb, C_rb, C_cb, coeff):
    
    products = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # All contribute to... well, M1 contributes to C11 and C22.
    # But in the tensor decomposition, each "product" M_s is itself a bilinear form.
    # M1 = (A11 + A22)(B11 + B22) expands to 4 block products.
    
    # Actually, I think the cleaner way is:
    # Each M_s is a bilinear form in A and B. We decompose M_s into rank-1 terms.
    # Then M_s contributes to certain C blocks with certain coefficients.
    
    # Let me just enumerate all the individual block-product contributions.
    
    # M1 = (A11 + A22)(B11 + B22)
    # Block products in M1:
    # A11*B11, A11*B22, A22*B11, A22*B22 (all with coeff +1)
    
    # M2 = (A21 + A22)B11
    # Block products: A21*B11, A22*B11 (coeff +1)
    
    # M3 = A11(B12 - B22)
    # Block products: A11*B12 (coeff +1), A11*B22 (coeff -1)
    
    # M4 = A22(B21 - B11)
    # Block products: A22*B21 (coeff +1), A22*B11 (coeff -1)
    
    # M5 = (A11 + A12)B22
    # Block products: A11*B22 (coeff +1), A12*B22 (coeff +1)
    
    # M6 = (A21 - A11)(B11 + B12)
    # Block products: A21*B11 (coeff +1), A21*B12 (coeff +1), A11*B11 (coeff -1), A11*B12 (coeff -1)
    
    # M7 = (A12 - A22)(B21 + B22)
    # Block products: A12*B21 (coeff +1), A12*B22 (coeff +1), A22*B21 (coeff -1), A22*B22 (coeff -1)
    
    # Now, each M_s contributes to C blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each block product in M_s, it contributes to each C block that M_s contributes to, with the product of the M_s-to-C coefficient and the block-product coefficient within M_s.
    
    # Let me build a list of (A_rb, A_cb, B_rb, B_cb, C_rb, C_cb, total_coeff) for each rank-1 contribution.
    
    # For each M_s, for each block product (A_rb, A_cb, B_rb, B_cb) with coeff c_mp:
    #   For each (C_rb, C_cb, c_mc) where M_s contributes to C_{C_rb,C_cb} with coeff c_mc:
    #     Add (A_rb, A_cb, B_rb, B_cb, C_rb, C_cb, c_mp * c_mc)
    
    # M1 contributes to: C11 (+1), C22 (+1)
    # M2 contributes to: C21 (+1), C22 (-1)
    # M3 contributes to: C12 (+1), C22 (+1)
    # M4 contributes to: C11 (+1), C21 (+1)
    # M5 contributes to: C11 (-1), C12 (+1)
    # M6 contributes to: C22 (+1)
    # M7 contributes to: C11 (+1)
    
    # Let me code this up.
    
    m_contributions = {
        1: [(0,0,1), (1,1,1)],       # M1 -> C11(+1), C22(+1)
        2: [(1,0,1), (1,1,-1)],      # M2 -> C21(+1), C22(-1)
        3: [(0,1,1), (1,1,1)],       # M3 -> C12(+1), C22(+1)
        4: [(0,0,1), (1,0,1)],       # M4 -> C11(+1), C21(+1)
        5: [(0,0,-1), (0,1,1)],      # M5 -> C11(-1), C12(+1)
        6: [(1,1,1)],                 # M6 -> C22(+1)
        7: [(0,0,1)],                 # M7 -> C11(+1)
    }
    
    # Block products within each M_s: (A_rb, A_cb, B_rb, B_cb, coeff)
    m_block_products = {
        1: [(0,0,0,0,1), (0,0,0,1,1), (1,1,0,0,1), (1,1,0,1,1)],
        2: [(1,0,0,0,1), (1,1,0,0,1)],
        3: [(0,0,0,1,1), (0,0,1,1,-1)],
        4: [(1,1,1,0,1), (1,1,0,0,-1)],
        5: [(0,0,1,1,1), (0,1,1,1,1)],
        6: [(1,0,0,0,1), (1,0,0,1,1), (0,0,0,0,-1), (0,0,0,1,-1)],
        7: [(0,1,1,0,1), (0,1,1,1,1), (1,1,1,0,-1), (1,1,1,1,-1)],
    }
    
    # Now build the full list of contributions
    contributions = []
    for ms in range(1, 8):
        for c_rb, c_cb, c_mc in m_contributions[ms]:
            for a_rb, a_cb, b_rb, b_cb, c_mp in m_block_products[ms]:
                total_coeff = c_mc * c_mp
                if total_coeff != 0:
                    contributions.append((a_rb, a_cb, b_rb, b_cb, c_rb, c_cb, total_coeff))
    
    # Now fill in U, V, W
    col = 0
    for (a_rb, a_cb, b_rb, b_cb, c_rb, c_cb, coeff) in contributions:
        # For each (j_local, k_local) in this block product:
        for j_local in range(2):
            for k_local in range(4):
                i = a_rb
                j_a = a_cb * 2 + j_local
                j_b = b_rb * 2 + j_local
                k = c_cb * 4 + k_local
                
                U[i * m + j_a][col] += coeff
                V[j_b * p + k][col] += coeff
                W[k * n + c_rb][col] += coeff
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
