"""Matrix multiplication 2x4 x 4x8 via tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Approach: Use the fact that M_{2,4,8} can be decomposed by treating the
    4x8 matrix as two 4x4 blocks, each requiring M_{2,4,4}.
    
    For M_{2,4,4}: split A (2x4) into two 2x2 blocks, B (4x4) into four 2x2 blocks.
    C = A_L * B_TL + A_L * B_TR + A_R * B_BL + A_R * B_BR
    Each 2x2*2x2 can use Strassen (rank 7), so 4*7 = 28 for M_{2,4,4}.
    For M_{2,4,8}: two such M_{2,4,4} instances, total rank = 56.
    
    However, we can do better by sharing some Strassen products.
    
    Alternative: Direct construction using bilinear forms.
    
    Let's use a more direct approach:
    A is 2x4, B is 4x8, C is 2x8.
    C[i,k] = sum_j A[i,j] * B[j,k] for j in 0..3
    
    We can write this as 8 independent 2x4*4x1 products... but that's not helpful.
    
    Better: Use the 8x8 matrix B and note that the tensor is:
    T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    
    Index: U has n*m = 8 rows, V has m*p = 32 rows, W has p*n = 16 rows.
    
    Let me construct this using a Kronecker product structure.
    
    Actually, the simplest correct approach:
    - M_{2,4,8} = M_{2,4,4} ⊗ M_{1,1,2} in some sense, but that gives rank 28*2=56.
    
    Let me try a different decomposition:
    Split B into 4 columns at a time: B = [B1, B2] where each Bi is 4x4.
    C = A*B1 + A*B2 where each term is 2x4.
    
    For A*B1 (2x4 * 4x4 = 2x4):
    Split A = [AL, AR] (each 2x2), B1 = [[B11, B12], [B21, B22]] (each 2x2).
    Result = AL*B11 + AL*B12 + AR*B21 + AR*B22 (each 2x2).
    4 Strassen instances, rank 28.
    
    Similarly for A*B2, another 28. Total 56.
    
    This matches the current program. Let me verify it's correct and clean it up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen for 2x2 * 2x2:
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # P0 = (a0+a3)*(b0+b3)
    # P1 = (a2+a3)*b0
    # P2 = a0*(b1-b3)
    # P3 = a3*(b2-b0)
    # P4 = (a0+a1)*b3
    # P5 = (a2-a0)*(b0+b1)
    # P6 = (a1-a3)*(b2+b3)
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # Define Strassen products: (A_factors, B_factors)
    # A_factors: list of (element_index, coefficient)
    # element_index: 0=a0, 1=a1, 2=a2, 3=a3
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P0
        ([(2, 1), (3, 1)], [(0, 1)]),           # P1
        ([(0, 1)], [(1, 1), (3, -1)]),          # P2
        ([(3, 1)], [(2, 1), (0, -1)]),          # P3
        ([(0, 1), (1, 1)], [(3, 1)]),           # P4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P6
    ]
    
    # C reconstruction: for each C element, list of (product_index, coefficient)
    c_recon = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]
    
    # 8 Strassen instances:
    # (a_col_off, b_row_off, b_col_off)
    # a_col_off: 0 (left half of A) or 2 (right half of A)
    # b_row_off: 0 (top half of B) or 2 (bottom half of B)  
    # b_col_off: 0, 2, 4, 6 (column offset in B)
    
    # For the left 4 columns of C (k=0..3):
    #   A_L * B[0:2, 0:2] -> (0, 0, 0)
    #   A_L * B[0:2, 2:4] -> (0, 0, 2)
    #   A_R * B[2:4, 0:2] -> (2, 2, 0)
    #   A_R * B[2:4, 2:4] -> (2, 2, 2)
    # For the right 4 columns of C (k=4..7):
    #   A_L * B[0:2, 4:6] -> (0, 0, 4)
    #   A_L * B[0:2, 6:8] -> (0, 0, 6)
    #   A_R * B[2:4, 4:6] -> (2, 2, 4)
    #   A_R * B[2:4, 6:8] -> (2, 2, 6)
    
    instances = [
        (0, 0, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
        (0, 0, 4),
        (0, 0, 6),
        (2, 2, 4),
        (2, 2, 6),
    ]
    
    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx][col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx][col] += b_coeff
            col += 1
    
    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_recon[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
