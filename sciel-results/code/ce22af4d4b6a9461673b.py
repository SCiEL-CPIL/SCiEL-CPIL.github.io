"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return the Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U, V, W matrices for the tensor decomposition.
    U: (4, 7) - linear combinations of A elements
    V: (4, 7) - linear combinations of B elements  
    W: (4, 7) - how to combine products to get C elements
    """
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    
    # Strassen's 7 products:
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    
    # U: rows correspond to a0,a1,a2,a3; columns to M1..M7
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # a0
        [0, 0, 0, 0, 1, 0, 1],    # a1
        [0, 1, 0, 0, 0, 1, 0],    # a2
        [1, 1, 0, 1, 0, 0, -1]    # a3
    ], dtype=float)
    
    # V: rows correspond to b0,b1,b2,b3; columns to M1..M7
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # b0
        [0, 0, 1, 0, 0, 1, 0],    # b1
        [0, 0, 0, 1, 0, 0, 1],    # b2
        [1, 0, -1, 0, 1, 0, 1]    # b3
    ], dtype=float)
    
    # W: rows correspond to c0,c1,c2,c3; columns to M1..M7
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c0 = C11
        [0, 0, 1, 0, 1, 0, 0],    # c1 = C12
        [0, 1, 0, 1, 0, 0, 0],    # c2 = C21
        [1, -1, 1, 0, 0, 1, 0]    # c3 = C22
    ], dtype=float)
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions to get a larger one.
    If dec1 multiplies (n1 x m1) by (m1 x p1) with rank r1,
    and dec2 multiplies (n2 x m2) by (m2 x p2) with rank r2,
    this doesn't directly apply. Instead, for Strassen recursion,
    we need to compose the linear maps.
    """
    # This function is not used directly; see construct() for the actual logic.
    pass

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: 
    # 1. Use Strassen for 4x4 x 4x4 part (first 4 cols of B), rank 49
    # 2. Use schoolbook for 4x4 x 1 part (last col of B), rank 16
    # Total rank: 65
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # For 4x4 x 4x4 using Strassen recursion:
    # Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B (4x4) into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Strassen products (each is a 2x2 x 2x2 multiplication):
    # S1 = (A11+A22) * (B11+B22)
    # S2 = (A21+A22) * B11
    # S3 = A11 * (B12-B22)
    # S4 = A22 * (B21-B11)
    # S5 = (A11+A12) * B22
    # S6 = (A21-A11) * (B11+B12)
    # S7 = (A12-A22) * (B21+B22)
    #
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Each Sj is a 2x2 x 2x2 multiplication, decomposable with rank 7.
    # So total rank for 4x4 x 4x4 is 7*7 = 49.
    
    # Let's build the U, V, W for the 4x4 x 4x4 part.
    # A is 4x4, indexed as A[i,j] for i,j in 0..3
    # B is 4x4, indexed as B[j,k] for j,k in 0..3
    # C is 4x4, indexed as C[i,k] for i,k in 0..3
    
    # U for 4x4 x 4x4: (16, 49)
    # V for 4x4 x 4x4: (16, 49)  
    # W for 4x4 x 4x4: (16, 49)
    
    # For each of the 7 Strassen products Sj, we have a 2x2 x 2x2 sub-decomposition.
    # The input to Sj is a 2x2 matrix (from linear combos of A blocks) and a 2x2 matrix (from linear combos of B blocks).
    # The output is a 2x2 matrix, which contributes to specific 2x2 blocks of C.
    
    # Let's define the 7 Strassen products and their contributions to C blocks.
    # Each product Sj produces a 2x2 result.
    
    # Product definitions (which blocks of A and B are combined):
    # S1: A_part = A11+A22, B_part = B11+B22
    # S2: A_part = A21+A22, B_part = B11
    # S3: A_part = A11, B_part = B12-B22
    # S4: A_part = A22, B_part = B21-B11
    # S5: A_part = A11+A12, B_part = B22
    # S6: A_part = A21-A11, B_part = B11+B12
    # S7: A_part = A12-A22, B_part = B21+B22
    
    # Contribution to C:
    # C11 (top-left 2x2) += S1 + S4 - S5 + S7
    # C12 (top-right 2x2) += S3 + S5
    # C21 (bottom-left 2x2) += S2 + S4
    # C22 (bottom-right 2x2) += S1 - S2 + S3 + S6
    
    # For each product Sj, the 2x2 x 2x2 decomposition has:
    # - U_j: (4, 7) mapping from the 4 elements of the A_part 2x2 matrix
    # - V_j: (4, 7) mapping from the 4 elements of the B_part 2x2 matrix
    # - W_j: (4, 7) mapping from the 7 products to the 4 elements of the 2x2 result
    
    # Now, the A_part for each Sj is a linear combination of A's 2x2 blocks.
    # Similarly for B_part.
    
    # Let me define the block indices:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # For each product Sj, I need to:
    # 1. Determine which A blocks are combined and with what coefficients
    # 2. Determine which B blocks are combined and with what coefficients
    # 3. Apply the 2x2 Strassen decomposition to the combined blocks
    
    # Let's build this step by step.
    
    # First, define the 7 products with their A and B block coefficients.
    # Each product uses a 2x2 matrix from A and a 2x2 matrix from B.
    
    # Product specs: (a_coeff, b_coeff) where a_coeff is a 4x4 matrix (coefficients for A blocks)
    # and b_coeff is a 4x4 matrix (coefficients for B blocks).
    # Actually, it's simpler to think of it as: for each product, which 2x2 blocks of A are added,
    # and which 2x2 blocks of B are added.
    
    # Let me use a different approach. I'll directly construct the U, V, W matrices.
    
    # For the 4x4 x 4x4 part:
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    
    # Let's build it product by product.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # For each of the 7 Strassen products at the top level:
    product_specs = [
        # (a_blocks, b_blocks) where a_blocks is a dict from (block_row, block_col) to coefficient
        # Block (0,0) = A11, (0,1) = A12, (1,0) = A21, (1,1) = A22
        # Similarly for B blocks.
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # S1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # S2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # S3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # S4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # S5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # S6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})    # S7
    ]
    
    # Contribution of each product to C blocks:
    # C block (0,0) = C11, (0,1) = C12, (1,0) = C21, (1,1) = C22
    c_contributions = [
        # For each product, which C blocks it contributes to and with what coefficient
        [(0,0, 1), (1,1, 1)],       # S1 -> C11 (+1), C22 (+1)
        [(1,0, 1), (1,1, -1)],      # S2 -> C21 (+1), C22 (-1)
        [(0,1, 1), (1,1, 1)],       # S3 -> C12 (+1), C22 (+1)
        [(0,0, 1), (1,0, 1)],       # S4 -> C11 (+1), C21 (+1)
        [(0,0, -1), (0,1, 1)],      # S5 -> C11 (-1), C12 (+1)
        [(1,1, 1)],                 # S6 -> C22 (+1)
        [(0,0, 1)]                  # S7 -> C11 (+1)
    ]
    
    # Wait, let me double-check the Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # S1 (M1) -> C11 (+1), C22 (+1)
    # S2 (M2) -> C21 (+1), C22 (-1)
    # S3 (M3) -> C12 (+1), C22 (+1)
    # S4 (M4) -> C11 (+1), C21 (+1)
    # S5 (M5) -> C11 (-1), C12 (+1)
    # S6 (M6) -> C22 (+1)
    # S7 (M7) -> C11 (+1)
    
    # That matches what I had above. Good.
    
    for prod_idx, (a_blocks, b_blocks) in enumerate(product_specs):
        # For this product, the A_part is a 2x2 matrix formed by linear combinations of A's 2x2 blocks.
        # The B_part is a 2x2 matrix formed by linear combinations of B's 2x2 blocks.
        
        # The 2x2 Strassen decomposition gives us U2, V2, W2 (each 4x7).
        # U2 maps the 4 elements of the 2x2 A_part to 7 linear combinations.
        # V2 maps the 4 elements of the 2x2 B_part to 7 linear combinations.
        # W2 maps the 7 products to the 4 elements of the 2x2 result.
        
        # Now, the 4 elements of the A_part 2x2 matrix are linear combinations of A's 16 elements.
        # Similarly for B_part.
        
        # Let's build the mapping from A's 16 elements to the 4 elements of A_part.
        # A is 4x4, indexed as A[i,j] for i,j in 0..3.
        # A_part is 2x2, with elements at positions (0,0), (0,1), (1,0), (1,1).
        
        # For each element (r,c) of A_part, it's a linear combination of A's 2x2 blocks.
        # a_blocks is a dict from (block_row, block_col) to coefficient.
        # Block (br, bc) corresponds to A[2*br:2*br+2, 2*bc:2*bc+2].
        
        # So A_part[r,c] = sum_{(br,bc), coeff in a_blocks} coeff * A[2*br+r, 2*bc+c]
        
        # Similarly for B_part.
        
        # Now, for the 7 sub-products in this Strassen product:
        for sub_idx in range(7):
            col = prod_idx * 7 + sub_idx
            
            # U mapping: A's 16 elements -> this sub-product's A linear combination
            # U2[sub_idx] gives the coefficients for the 4 elements of A_part.
            for r in range(2):
                for c in range(2):
                    a_part_elem = r * 2 + c  # index in 0..3 for the 4 elements of A_part
                    coeff_a = U2[a_part_elem, sub_idx]
                    if coeff_a != 0:
                        # A_part[r,c] = sum_{(br,bc), coeff in a_blocks} coeff * A[2*br+r, 2*bc+c]
                        for (br, bc), block_coeff in a_blocks.items():
                            i = 2 * br + r
                            j = 2 * bc + c
                            U_44[i * 4 + j, col] += block_coeff * coeff_a
            
            # V mapping: B's 16 elements -> this sub-product's B linear combination
            for r in range(2):
                for c in range(2):
                    b_part_elem = r * 2 + c
                    coeff_b = V2[b_part_elem, sub_idx]
                    if coeff_b != 0:
                        for (br, bc), block_coeff in b_blocks.items():
                            j = 2 * br + r
                            k = 2 * bc + c
                            V_44[j * 4 + k, col] += block_coeff * coeff_b
            
            # W mapping: this sub-product's 2x2 result -> C's 16 elements
            # The 2x2 result has elements at positions (r,c) for r,c in 0..1.
            # W2[r*2+c, sub_idx] gives the coefficient for this element in the sub-product.
            for r in range(2):
                for c in range(2):
                    result_elem = r * 2 + c
                    coeff_w = W2[result_elem, sub_idx]
                    if coeff_w != 0:
                        # This element contributes to C blocks based on c_contributions[prod_idx]
                        for (cbr, cbc, c_coeff) in c_contributions[prod_idx]:
                            i = 2 * cbr + r
                            k = 2 * cbc + c
                            W_44[k * 4 + i, col] += c_coeff * coeff_w
    
    # Now I have U_44, V_44, W_44 for the 4x4 x 4x4 part (rank 49).
    
    # For the 4x4 x 1 part (last column of B):
    # A is 4x4, B_last is 4x1 (column vector).
    # C_last is 4x1, where C_last[i,0] = sum_j A[i,j] * B_last[j,0]
    # This is 16 rank-1 terms (one for each (i,j) pair).
    
    U_41 = np.zeros((16, 16))
    V_41 = np.zeros((4, 16))
    W_41 = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_41[i * 4 + j, col] = 1
            V_41[j, col] = 1
            W_41[i, col] = 1
            col += 1
    
    # Combine the two decompositions:
    # Total U: (16, 49+16) = (16, 65)
    # Total V: (16+4, 49+16) = (20, 65)  -- wait, V should be (m*p, rank) = (4*5, 65) = (20, 65)
    # Total W: (p*n, rank) = (5*4, 65) = (20, 65)
    
    # For the 4x4 x 4x4 part:
    # U_44: (16, 49) -- A is 4x4, so n*m = 16
    # V_44: (16, 49) -- B is 4x4 for this part, so m*p = 16
    # W_44: (16, 49) -- C is 4x4 for this part, so p*n = 16
    
    # For the 4x4 x 1 part:
    # U_41: (16, 16) -- A is 4x4
    # V_41: (4, 16) -- B_last is 4x1, so m*p = 4*1 = 4
    # W_41: (4, 16) -- C_last is 4x1, so p*n = 1*4 = 4
    
    # Now, the full B is 4x5, so V should have 4*5 = 20 rows.
    # The first 16 rows correspond to B[0:4, 0:4] (the 4x4 part).
    # The last 4 rows correspond to B[0:4, 4] (the 1-column part).
    
    # The full C is 4x5, so W should have 5*4 = 20 rows.
    # The first 16 rows correspond to C[0:4, 0:4] (the 4x4 part).
    # The last 4 rows correspond to C[0:4, 4] (the 1-column part).
    
    # Let me assemble the full U, V, W.
    
    rank = 65
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # 4x4 x 4x4 part (columns 0..48)
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # 4x4 x 1 part (columns 49..64)
    U[:, 49:65] = U_41
    V[16:20, 49:65] = V_41
    W[16:20, 49:65] = W_41
    
    # Convert to lists of lists for the return format.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify the decomposition (optional, for debugging)
    # For random A and B, check that sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] == C[i',k']
    # where C = A @ B
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return what we have
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

