import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # Block decomposition: 3x3 blocks of 2x2
    block_size = 2
    n_blocks = 3

    # Rank = 23 (Stothers for 3x3) * 7 (Strassen for 2x2) = 161
    rank = 23 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 7 terms for 2x2 blocks
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Stothers 23 terms for 3x3 blocks
    # Coefficients for A and B blocks
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    
    # The 23 linear combinations (coefficients for A00..A22, B00..B22)
    # Format: (list of (i, j, coeff) for A, list of (i, j, coeff) for B)
    stothers_terms = [
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
        ([(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,-1)]),
        ([(0,1,1), (1,1,-1), (2,1,1)], [(1,0,1), (1,1,1), (1,2,-1)]),
        ([(0,2,1), (1,2,-1), (2,2,1)], [(2,0,1), (2,1,1), (2,2,-1)]),
        ([(0,0,1), (0,1,1), (1,1,1), (1,2,1), (2,2,1)], [(0,0,1), (0,2,1), (1,0,1), (1,2,1), (2,0,1), (2,2,1)]),
        ([(0,0,1), (0,1,-1), (1,0,1), (1,1,-1), (2,0,1), (2,1,-1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,0,1), (2,1,1)]),
        ([(0,0,1), (0,2,1), (1,0,1), (1,2,1), (2,0,1), (2,2,1)], [(0,0,1), (0,2,1), (1,0,1), (1,2,1), (2,0,1), (2,2,1)]),
        ([(0,1,1), (1,2,1), (2,0,1)], [(0,1,1), (1,2,1), (2,0,1)]),
        ([(0,2,1), (1,0,1), (2,1,1)], [(0,2,1), (1,0,1), (2,1,1)]),
        ([(0,1,1), (1,0,1), (2,2,1)], [(0,1,1), (1,0,1), (2,2,1)]),
        ([(0,2,1), (1,1,1), (2,0,1)], [(0,2,1), (1,1,1), (2,0,1)]),
        ([(0,0,1), (1,2,1), (2,1,1)], [(0,0,1), (1,2,1), (2,1,1)]),
        ([(0,1,1), (1,2,-1), (2,0,1)], [(0,1,1), (1,2,-1), (2,0,1)]),
        ([(0,2,1), (1,0,-1), (2,1,1)], [(0,2,1), (1,0,-1), (2,1,1)]),
        ([(0,0,1), (1,1,-1), (2,2,-1)], [(0,0,1), (1,1,-1), (2,2,-1)]),
        ([(0,1,1), (1,1,-1), (2,1,-1)], [(1,0,1), (1,1,-1), (1,2,-1)]),
        ([(0,2,1), (1,2,-1), (2,2,-1)], [(2,0,1), (2,1,-1), (2,2,-1)]),
        ([(0,0,1), (0,1,1), (1,1,-1), (1,2,-1), (2,2,1)], [(0,0,1), (0,2,1), (1,0,-1), (1,2,-1), (2,0,1), (2,2,1)]),
        ([(0,0,1), (0,1,-1), (1,0,1), (1,1,1), (2,0,-1), (2,1,-1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1), (2,0,1), (2,1,1)]),
        ([(0,0,1), (0,2,1), (1,0,-1), (1,2,-1), (2,0,1), (2,2,1)], [(0,0,1), (0,2,1), (1,0,1), (1,2,-1), (2,0,1), (2,2,1)]),
        ([(0,1,1), (1,2,1), (2,0,-1)], [(0,1,1), (1,2,1), (2,0,-1)]),
        ([(0,2,1), (1,0,1), (2,1,-1)], [(0,2,1), (1,0,1), (2,1,-1)]),
        ([(0,1,1), (1,0,-1), (2,2,-1)], [(0,1,1), (1,0,-1), (2,2,-1)]),
    ]

    # Iterate over the 23 Stothers terms
    for a_coeffs, b_coeffs in stothers_terms:
        # For each Stothers term, we perform a 2x2 block multiplication using Strassen
        # This adds 7 rank-1 terms to the global decomposition
        
        # M1 = (X00 + X11)(Y00 + Y11)
        # X00, X11 etc are 2x2 blocks.
        # We need to add the Strassen terms for the product of the linear combinations.
        
        # Helper to add Strassen terms for a given pair of linear combinations
        def add_strassen(a_list, b_list):
            nonlocal col
            # M1
            # U: (X00 + X11)
            for ib, kb, coeff in a_list:
                if coeff == 1:
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    U[u_idx(ib, kb, 1, 1), col] += 1
                elif coeff == -1:
                    U[u_idx(ib, kb, 0, 0), col] -= 1
                    U[u_idx(ib, kb, 1, 1), col] -= 1
            # V: (Y00 + Y11)
            for kb, jb, coeff in b_list:
                if coeff == 1:
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                elif coeff == -1:
                    V[v_idx(kb, jb, 0, 0), col] -= 1
                    V[v_idx(kb, jb, 1, 1), col] -= 1
            # W: (Z00 + Z11)
            for ib, jb, coeff in a_list: # Wait, W depends on the result block
                # The result of the Stothers term contributes to the block product.
                # We need to map the result of the 2x2 block multiplication to the 3x3 block structure.
                # The Stothers term calculates a specific linear combination of the block products.
                # Actually, the Stothers algorithm calculates C = A B.
                # Each of the 23 terms is a product L_A L_B.
                # L_A and L_B are 3x3 matrices of 2x2 blocks.
                # The product L_A L_B is a 3x3 matrix of 2x2 blocks.
                # We need to decompose this product.
                # But the product is not a single 2x2 block multiplication.
                # It is a 3x3 matrix multiplication of 2x2 blocks.
                # This is circular.
                
                pass

    # The above logic is flawed. Stothers decomposes the 3x3 multiplication into 23 scalar multiplications.
    # If the "scalars" are 2x2 matrices, then each "scalar multiplication" is a 2x2 matrix multiplication.
    # So we need to decompose 23 instances of 2x2 matrix multiplication.
    # Each 2x2 matrix multiplication is decomposed by Strassen (7 terms).
    # Total 23 * 7 = 161 terms.
    
    # The 23 terms are products of linear combinations of the 2x2 blocks.
    # Let L_A be a 3x3 matrix of 2x2 blocks.
    # Let L_B be a 3x3 matrix of 2x2 blocks.
    # The product L_A L_B is a 3x3 matrix of 2x2 blocks.
    # We need to add the contributions of this product to the final C.
    # But the product L_A L_B is not a single 2x2 block. It is a 3x3 matrix of 2x2 blocks.
    # So we cannot just apply Strassen to a single 2x2 block.
    
    # Correct interpretation:
    # The Stothers algorithm expresses the product C = A B as a sum of 23 terms P_t.
    # P_t = L_A^{(t)} L_B^{(t)}.
    # L_A^{(t)} and L_B^{(t)} are 3x3 matrices whose entries are linear combinations of the 2x2 blocks of A and B.
    # So P_t is a 3x3 matrix whose entries are products of linear combinations of 2x2 blocks.
    # Each entry of P_t is a sum of products of 2x2 blocks.
    # This is not a single 2x2 block multiplication.
    
    # Therefore, the rank is not 23 * 7.
    # The rank is the sum of the ranks of the 23 terms.
    # Each term P_t is a 3x3 matrix of 2x2 blocks.
    # The entries of P_t are linear combinations of products of 2x2 blocks.
    # This seems complicated.
    
    # Alternative: Use the fact that R(6) <= R(2) R(3) is for the tensor product.
    # The tensor product of the 2x2 multiplication tensor and the 3x3 multiplication tensor gives a 6x6 multiplication tensor.
    # The rank of the tensor product is the product of the ranks.
    # So R(6) <= R(2) * R(3) = 7 * 23 = 161.
    # This construction is valid.
    # The 161 terms are formed by taking each of the 7 Strassen terms for 2x2 and each of the 23 Stothers terms for 3x3.
    # Each pair (s, t) gives a rank-1 term for the 6x6 multiplication.
    
    # So I will iterate over the 7 Strassen terms and the 23 Stothers terms.
    # For each pair, I will construct the U, V, W vectors.
    
    pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
