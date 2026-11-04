"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 72 decomposition:
    # 1. Block Strassen for 4x4x4 part (using schoolbook for 2x2 blocks to ensure correctness)
    #    7 block multiplications, each rank 8 (schoolbook 2x2x2). Total 56.
    # 2. Schoolbook for 4x4x1 part. Rank 16.
    # Total rank 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using block schoolbook (rank 56)
    # We simulate the 7 Strassen block products, but expand each 2x2x2 into 8 scalar products.
    # This is equivalent to using schoolbook for the 2x2 blocks.
    # Actually, simpler: Just use schoolbook for the whole 4x4x4?
    # No, schoolbook 4x4x4 is rank 64.
    # Block schoolbook is 7 * 8 = 56.
    
    # Let's manually construct the 56 terms.
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pi is a 2x2x2 tensor.
    # We can just use the schoolbook decomposition for each 2x2x2 tensor.
    # There are 7 such tensors.
    # Each 2x2x2 schoolbook has 8 terms.
    # Total 56 terms.
    
    col = 0
    # Define the 7 block products as linear combinations of A and B blocks.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # For each of the 7 products, we define the linear forms for A and B.
    # Then we expand the 2x2 multiplication.
    
    # P1: A_form = A00+A11, B_form = B00+B11
    # P2: A_form = A10+A11, B_form = B00
    # P3: A_form = A00, B_form = B01-B11
    # P4: A_form = A11, B_form = B10-B00
    # P5: A_form = A00+A01, B_form = B11
    # P6: A_form = A10-A00, B_form = B00+B01
    # P7: A_form = A01-A11, B_form = B10+B11
    
    # We need to map these to the global indices.
    # Global A index: i*m + j.
    # Global B index: j*p + k.
    # Global W index: k*n + i.
    
    # For each of the 7 block products, we iterate through the 8 scalar multiplications of 2x2x2.
    # The 2x2x2 schoolbook terms are:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # corresponding to A[i,j] B[j,k] contributing to C[i,k].
    
    # We define the coefficients for A and B for each of the 7 products.
    # Let's create a list of (coeff_A, coeff_B) for each of the 4 elements of the 2x2 block.
    
    # Product 1:
    # A_block = [[1, 0], [0, 1]] (A00 + A11)
    # B_block = [[1, 0], [0, 1]] (B00 + B11)
    # Product 2:
    # A_block = [[0, 0], [1, 1]] (A10 + A11)
    # B_block = [[1, 0], [0, 0]] (B00)
    # Product 3:
    # A_block = [[1, 0], [0, 0]] (A00)
    # B_block = [[0, 1], [0, -1]] (B01 - B11)
    # Product 4:
    # A_block = [[0, 0], [0, 1]] (A11)
    # B_block = [[-1, 0], [1, 0]] (B10 - B00)
    # Product 5:
    # A_block = [[1, 1], [0, 0]] (A00 + A01)
    # B_block = [[0, 0], [0, 1]] (B11)
    # Product 6:
    # A_block = [[-1, 0], [1, 0]] (A10 - A00)
    # B_block = [[1, 1], [0, 0]] (B00 + B01)
    # Product 7:
    # A_block = [[0, 1], [0, -1]] (A01 - A11)
    # B_block = [[0, 0], [1, 1]] (B10 + B11)
    
    # Note: The indices in the blocks are local (0,1).
    # We need to map local (li, lj) to global (gi, gj).
    # Block 00: gi=0, gj=0
    # Block 01: gi=0, gj=1
    # Block 10: gi=1, gj=0
    # Block 11: gi=1, gj=1
    
    # For each product p_idx (0..6), and each local term (li, lj, lk) in 2x2x2 schoolbook:
    # The term is A_block[li, lj] * B_block[lj, lk] * W_block[lk, li]
    # Wait, the output block is also 2x2.
    # The contribution to C_block[lk, li] is A_block[li, lj] B_block[lj, lk].
    # So for each local (li, lj, lk), we add:
    # U[global_A(li, lj), col] += A_block[li, lj]
    # V[global_B(lj, lk), col] += B_block[lj, lk]
    # W[global_W(lk, li), col] += 1  (since C_block[lk, li] gets this term)
    
    # Wait, the W matrix accumulates the contribution to the output.
    # In the tensor sum, W is just a coefficient.
    # The term is U * V * W.
    # If U and V are the coefficients of A and B, then W must be 1 for the output element.
    # But we are summing 7 products.
    # The output C is the sum of the 7 products' outputs.
    # So for each product, we add its contribution to the output.
    # So W should be 1 for the relevant output element.
    
    # Let's refine the loop.
    products = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]), # P1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]), # P2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]), # P3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]]), # P4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]), # P5
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]]), # P6
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]])  # P7
    ]
    
    for A_block, B_block in products:
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    # Local term A[li, lj] B[lj, lk] contributes to C[lk, li]
                    # Global indices:
                    gi = li
                    gj = lj
                    gk = lk
                    # A index: gi * m + gj
                    a_idx = gi * m + gj
                    # B index: gj * p + gk
                    # Note: B is 4x5. The blocks are 2x2.
                    # B_block[0,0] is B[0,0], B[0,1], B[1,0], B[1,1].
                    # So gj and gk are local indices 0,1.
                    # Global B index: gj * p + gk.
                    b_idx = gj * p + gk
                    # W index: gk * n + gi
                    w_idx = gk * n + gi
                    
                    u_val = A_block[li][lj]
                    v_val = B_block[lj][lk]
                    
                    if u_val != 0 and v_val != 0:
                        U[a_idx][col] += u_val
                        V[b_idx][col] += v_val
                        W[w_idx][col] += 1
                    col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    # Computes C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + 4
            w_idx = 4 * n + i
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
