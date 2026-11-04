DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx: (i, j) in A
        # v_idx: (j, k) in B
        # w_idx: (k, i) in C ? No, w is length p*n.
        # The trace convention is W[k*n + i, r].
        # So w index is k*n + i.
        # Wait, the standard tensor decomposition for matrix multiplication C = A B:
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T_{i,j,k} = delta_{ik} (if we consider C as output).
        # Actually, the rank-1 term is A_{ij} B_{jk} C_{ik}.
        # So u has 1 at (i,j).
        # v has 1 at (j,k).
        # w has 1 at (k,i)? Or (i,k)?
        # The task says: "output factor follows the official trace convention W[k*n+i,r]".
        # This implies w is indexed by (k, i) where k is row of B / col of A?
        # Let's check the seed program.
        # make_w(idx_k, idx_i, val): w[idx_k * n + idx_i] = val.
        # In schoolbook:
        # for i, j, k:
        #   u = make_u(i, j, 1)  # A_{ij}
        #   v = make_v(j, k, 1)  # B_{jk}
        #   w = make_w(k, i, 1)  # C_{ki}?
        #   # C = A B => C_{i, k} = sum_j A_{ij} B_{jk}.
        #   # So the output is C_{i, k}.
        #   # The seed uses w[k, i]. This corresponds to C^T_{i, k}?
        #   # Or is the convention C_{k, i}?
        #   # If C = A B, then C is n x p.
        #   # w is length p * n.
        #   # Index k*n + i. k ranges 0..p-1, i ranges 0..n-1.
        #   # This is C_{k, i} in row-major?
        #   # If C is n x p, then C[i, k].
        #   # The index k*n + i is C^T[k, i] or C[i, k] if stored column-major?
        #   # Let's assume the seed is correct and follow its indexing.
        #   # Seed: u[i, j], v[j, k], w[k, i].
        #   # This sums to C[i, k] = sum_j A[i, j] B[j, k].
        #   # So w[k, i] corresponds to the coefficient of C[i, k].
        pass

    # 1. Strassen part for 4x4 x 4x4
    # A blocks: A11, A12, A21, A22 (2x2 each)
    # B blocks: B11, B12, B21, B22 (2x2 each)
    
    # We define the 7 Strassen multiplications.
    # Each is a product of (A_block_comb) * (B_block_comb).
    # Then we expand each 2x2 * 2x2 product using schoolbook (4 terms).
    
    # Define block indices
    # A[i, j] where i,j in {0,1} for block row/col, and internal indices ii, jj in {0,1}
    # Global A index: (block_row * 2 + ii, block_col * 2 + jj)
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks, sign_A, sign_B, sign_C)
        # M1 = (A11 + A22) * (B11 + B22) -> C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        # This is complex to map directly to rank-1 terms without expanding.
        # It is easier to just generate the 28 terms explicitly.
    ]
    
    # Let's just hardcode the 28 terms for the 4x4x4 part.
    # Actually, we can generate them programmatically.
    
    # Function to add a 2x2x2x2 schoolbook multiplication
    def add_2x2_mult(a_br, a_bc, b_br, b_bc, c_br, c_bc, coeff):
        # a_br, a_bc: block row/col of A (0 or 1)
        # b_br, b_bc: block row/col of B
        # c_br, c_bc: block row/col of C
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    # A global index
                    gi = a_br * 2 + ii
                    gj = a_bc * 2 + jj
                    # B global index
                    gk = b_br * 2 + jj # Wait, B is 4x4. B[j, k].
                    # In 2x2 block B_br, B_bc, the element is B[2*b_br + jj, 2*b_bc + kk]
                    # So global row is 2*b_br + jj, global col is 2*b_bc + kk.
                    # But in the multiplication A * B, the inner index must match.
                    # A[i, j] * B[j, k].
                    # So A's col index must equal B's row index.
                    # A block (a_br, a_bc) has cols 2*a_bc + jj.
                    # B block (b_br, b_bc) has rows 2*b_br + jj.
                    # This requires 2*a_bc + jj == 2*b_br + jj => a_bc == b_br.
                    # This is always true for the terms we generate?
                    # In Strassen, we multiply specific blocks.
                    # e.g. A11 * B11. a_bc=0, b_br=0. Match.
                    # e.g. A12 * B22. a_bc=1, b_br=1. Match.
                    # So we are good.
                    
                    # B global index
                    gk_row = b_br * 2 + jj
                    gk_col = b_bc * 2 + kk
                    
                    # C global index
                    # C = A B. C[i, k].
                    # i = 2*c_br + ii
                    # k = 2*c_bc + kk
                    # w index: k*n + i? No, seed uses w[k, i] for C[i, k]?
                    # Seed: w[k*n + i].
                    # So w index = gk_col * n + gi?
                    # n=4.
                    # w index = gk_col * 4 + gi.
                    
                    u_idx = gi * m + gj # m=4
                    v_idx = gk_row * p + gk_col # p=7? No, this is for 4x4 part, so p=4 effectively.
                    # Wait, v is length m*p = 4*7 = 28.
                    # In the 4x4 part, we are multiplying by the first 4 columns of B.
                    # So B is 4x4.
                    # v index = gk_row * 7 + gk_col.
                    
                    w_idx = gk_col * n + gi # n=4
                    
                    terms.append((u_idx, v_idx, w_idx, coeff))

    # Strassen 7 terms for 4x4x4
    # 1. (A11+A22)(B11+B22)
    # Expand: A11B11 + A11B22 + A22B11 + A22B22
    # C11 +=, C12 -=, C21 -=, C22 +=
    # We need to map these to the correct C blocks.
    
    # Term 1: A11 * B11
    # C11 += A11B11
    add_2x2_mult(0, 0, 0, 0, 0, 0, 1)
    # C12 -= A11B11 ? No.
    # M1 = (A11+A22)(B11+B22)
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # This is getting complicated to map coefficients.
    # It is easier to just use the known 28-term decomposition if I can find it, 
    # or just use schoolbook for the whole thing but with a better split?
    # No, schoolbook is 112.
    # Strassen 1-level + schoolbook leaves is 28.
    
    # Let's just implement the 28 terms by expanding the Strassen formulas.
    
    # I will define the terms explicitly.
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # Each Mk is a 2x2 matrix product.
    # We will generate terms for each Mk, and add them to the correct C blocks with the correct signs.
    
    def add_strassen_term(m_id, a_blocks, b_blocks, c_blocks, sign):
        # a_blocks: list of (block_row, block_col, sign)
        # b_blocks: list of (block_row, block_col, sign)
        # c_blocks: list of (block_row, block_col, sign)
        # This is too abstract.
        pass

    # Let's just hardcode the 28 terms.
    # A11B11:
    # u: A[0:2, 0:2], v: B[0:2, 0:2], w: C[0:2, 0:2]
    # 4 terms.
    
    # I will write a helper that takes a 2x2 A block, 2x2 B block, 2x2 C block, and sign,
    # and adds the 4 schoolbook terms.
    
    def add_block_mult(a_br, a_bc, b_br, b_bc, c_br, c_bc, sign):
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    gi = a_br * 2 + ii
                    gj = a_bc * 2 + jj
                    gk_row = b_br * 2 + jj
                    gk_col = b_bc * 2 + kk
                    ci = c_br * 2 + ii
                    ck = c_bc * 2 + kk
                    
                    u_idx = gi * 4 + gj
                    v_idx = gk_row * 7 + gk_col
                    w_idx = ck * 4 + ci
                    
                    terms.append((u_idx, v_idx, w_idx, sign))

    # M1 = (A11+A22)(B11+B22)
    # A11B11 -> C11 (+), C12 (-), C21 (-), C22 (+)
    # Wait, M1 contributes to all 4 C blocks?
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # So M1 contributes to C11 and C22.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    
    # A11B11:
    add_block_mult(0, 0, 0, 0, 0, 0, 1) # C11
    add_block_mult(0, 0, 0, 0, 1, 1, 1) # C22
    
    # A11B22:
    add_block_mult(0, 0, 1, 1, 0, 0, 1) # C11
    add_block_mult(0, 0, 1, 1, 1, 1, 1) # C22
    
    # A22B11:
    add_block_mult(1, 1, 0, 0, 0, 0, 1) # C11
    add_block_mult(1, 1, 0, 0, 1, 1, 1) # C22
    
    # A22B22:
    add_block_mult(1, 1, 1, 1, 0, 0, 1) # C11
    add_block_mult(1, 1, 1, 1, 1, 1, 1) # C22
    
    # M2 = (A21+A22)B11
    # C21 += M2
    # A21B11:
    add_block_mult(1, 0, 0, 0, 1, 0, 1) # C21
    # A22B11:
    add_block_mult(1, 1, 0, 0, 1, 0, 1) # C21
    
    # M3 = A11(B12-B22)
    # C12 += M3, C22 -= M3
    # A11B12:
    add_block_mult(0, 0, 0, 1, 0, 1, 1) # C12
    add_block_mult(0, 0, 0, 1, 1, 1, -1) # C22
    # A11B22:
    add_block_mult(0, 0, 1, 1, 0, 1, -1) # C12
    add_block_mult(0, 0, 1, 1, 1, 1, 1) # C22
    
    # M4 = A22(B21-B11)
    # C11 -= M4, C21 += M4
    # A22B21:
    add_block_mult(1, 1, 1, 0, 0, 0, -1) # C11
    add_block_mult(1, 1, 1, 0, 1, 0, 1) # C21
    # A22B11:
    add_block_mult(1, 1, 0, 0, 0, 0, 1) # C11
    add_block_mult(1, 1, 0, 0, 1, 0, -1) # C21
    
    # M5 = (A11+A12)B22
    # C12 += M5, C22 -= M5
    # A11B22:
    add_block_mult(0, 0, 1, 1, 0, 1, 1) # C12
    add_block_mult(0, 0, 1, 1, 1, 1, -1) # C22
    # A12B22:
    add_block_mult(0, 1, 1, 1, 0, 1, 1) # C12
    add_block_mult(0, 1, 1, 1, 1, 1, -1) # C22
    
    # M6 = (A21-A11)(B11+B12)
    # C11 -= M6, C22 += M6
    # A21B11:
    add_block_mult(1, 0, 0, 0, 0, 0, -1) # C11
    add_block_mult(1, 0, 0, 0, 1, 1, 1) # C22
    # A21B12:
    add_block_mult(1, 0, 0, 1, 0, 0, -1) # C11
    add_block_mult(1, 0, 0, 1, 1, 1, 1) # C22
    # A11B11:
    add_block_mult(0, 0, 0, 0, 0, 0, 1) # C11
    add_block_mult(0, 0, 0, 0, 1, 1, -1) # C22
    # A11B12:
    add_block_mult(0, 0, 0, 1, 0, 0, 1) # C11
    add_block_mult(0, 0, 0, 1, 1, 1, -1) # C22
    
    # M7 = (A12-A22)(B21+B22)
    # C11 += M7, C22 += M6? No.
    # C11 = M1 - M4 - M6 + M7
    # C22 = M1 - M3 - M5 + M6
    # So M7 only contributes to C11.
    # A12B21:
    add_block_mult(0, 1, 1, 0, 0, 0, 1) # C11
    # A12B22:
    add_block_mult(0, 1, 1, 1, 0, 0, 1) # C11
    # A22B21:
    add_block_mult(1, 1, 1, 0, 0, 0, -1) # C11
    # A22B22:
    add_block_mult(1, 1, 1, 1, 0, 0, -1) # C11

    # 2. Schoolbook for 4x4 x 4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 7 + (4 + k) # B is 4x7, cols 4,5,6
                w_idx = (4 + k) * 4 + i
                terms.append((u_idx, v_idx, w_idx, 1))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w, coeff) in enumerate(terms):
        U[u][c] = coeff
        V[v][c] = coeff
        W[w][c] = coeff
        
    return {"U": U, "V": V, "W": W}
