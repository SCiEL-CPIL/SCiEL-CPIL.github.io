"""Program for matrix multiplication tensor rank minimization."""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Build an exact rank-56 decomposition of the 2x2 @ 2x16 matrix multiplication
    tensor by grouping the 16 columns of B into 8 blocks of 2 columns each, and
    applying Laderman's rank-7 algorithm for 2x2 @ 2x2 to each block.

    For each block b (0..7), B[:, 2b:2b+2] is a 2x2 matrix. The product
    C[:, 2b:2b+2] = A @ B[:, 2b:2b+2] is a 2x2 product, which Laderman's
    algorithm computes with 7 rank-1 terms. Total rank = 8 * 7 = 56.

    Laderman's rank-7 decomposition for C = A @ B (2x2):
      t0 = A00 * B00
      t1 = A01 * B11
      t2 = A10 * B01
      t3 = (A00 + A11)(B00 + B11)
      t4 = (A01 + A11)(B10 + B11)
      t5 = (A10 + A00)(B01 + B00)
      t6 = (A10 + A11)(B10 + B11)
      
      C00 = t0 + t3 - t4 - t5
      C01 = t1 + t4 - t3 - t6
      C10 = t2 + t5 - t3 - t6
      C11 = t6 + t3 - t0 - t1 - t2
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # U: shape (n*m, rank) = (4, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (32, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's coefficients for the 7 terms
    # Each term is (U_coefficients for A, V_coefficients for B, W_coefficients for C)
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 01->1, 10->2, 11->3
    
    # Term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # W_coeffs are the contributions to C00, C01, C10, C11
    laderman_terms = [
        # t0 = A00 * B00
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),
        # t1 = A01 * B11
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # t2 = A10 * B01
        ([0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]),
        # t3 = (A00 + A11)(B00 + B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1]),
        # t4 = (A01 + A11)(B10 + B11)
        ([0, 1, 0, 1], [0, 0, 1, 1], [-1, 1, 0, 0]),
        # t5 = (A10 + A00)(B01 + B00)
        ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 0]),
        # t6 = (A10 + A11)(B10 + B11)
        ([0, 0, 1, 1], [0, 0, 1, 1], [0, -1, -1, 1]),
    ]
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            u_coeffs, v_coeffs, w_coeffs = laderman_terms[t]
            
            # U: A[i,j] -> row i*m + j
            for ai in range(4):
                U[ai][col] = u_coeffs[ai]
            
            # V: B[j, 2b+di] where di in {0,1}
            # B row index: j * p + (2b + di)
            # v_coeffs indices: 0->B00, 1->B01, 2->B10, 3->B11
            # B00: j=0, col=2b+0 -> row 0*16 + 2b = 2b
            # B01: j=0, col=2b+1 -> row 0*16 + 2b+1 = 2b+1
            # B10: j=1, col=2b+0 -> row 1*16 + 2b = 16+2b
            # B11: j=1, col=2b+1 -> row 1*16 + 2b+1 = 16+2b+1
            v_row_map = [2*b, 2*b+1, 16+2*b, 16+2*b+1]
            for vi in range(4):
                V[v_row_map[vi]][col] = v_coeffs[vi]
            
            # W: C[i, 2b+di] -> row (2b+di)*n + i
            # w_coeffs indices: 0->C00, 1->C01, 2->C10, 3->C11
            # C00: i=0, col=2b+0 -> row (2b+0)*2 + 0 = 4b
            # C01: i=0, col=2b+1 -> row (2b+1)*2 + 0 = 4b+2
            # C10: i=1, col=2b+0 -> row (2b+0)*2 + 1 = 4b+1
            # C11: i=1, col=2b+1 -> row (2b+1)*2 + 1 = 4b+3
            w_row_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
            for wi in range(4):
                W[w_row_map[wi]][col] = w_coeffs[wi]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
