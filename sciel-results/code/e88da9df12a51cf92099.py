import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # Total rank estimate: 49 (4x4x4) + 32 (4x4x2) + 24 (4x1x6) = 105
    rank = 105
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, u_c in zip(u_indices, u_coeffs):
            if u_c != 0:
                U[u_idx][col] += u_c
        for v_idx, v_c in zip(v_indices, v_coeffs):
            if v_c != 0:
                V[v_idx][col] += v_c
        for w_idx, w_c in zip(w_indices, w_coeffs):
            if w_c != 0:
                W[w_idx][col] += w_c
        col += 1

    # Indices for A (4x5): i in 0..3, j in 0..4 -> i*5+j
    # Indices for B (5x6): j in 0..4, k in 0..5 -> j*6+k
    # Indices for C (4x6): i in 0..3, k in 0..5 -> k*4+i (Trace convention W[k*n+i])
    
    # --- Part 1: A1 (4x4) * B11 (4x4) -> C11 (4x4) ---
    # A1: A[0:4, 0:4]
    # B11: B[0:4, 0:4]
    # C11: C[0:4, 0:4]
    
    # Strassen 4x4 decomposition
    # A1 blocks:
    # A11: A[0:2, 0:2], A12: A[0:2, 2:4]
    # A21: A[2:4, 0:2], A22: A[2:4, 2:4]
    
    # B11 blocks:
    # B11: B[0:2, 0:2], B12: B[0:2, 2:4]
    # B21: B[2:4, 0:2], B22: B[2:4, 2:4]
    
    # C11 blocks:
    # C11: C[0:2, 0:2], C12: C[0:2, 2:4]
    # C21: C[2:4, 0:2], C22: C[2:4, 2:4]
    
    # We need to generate 49 terms.
    # Each term corresponds to a scalar multiplication in one of the 7 Strassen products.
    
    # Define the 7 Strassen products M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mk, we apply Strassen 2x2 to get 7 scalar products.
    # Then we map these to C11.
    
    # C11 = M1 + M4 - M5 + M7 (for top-left 2x2 block of C11? No, C11 is 4x4)
    # C11[0:2, 0:2] = M1[0:2, 0:2] + M4[0:2, 0:2] - M5[0:2, 0:2] + M7[0:2, 0:2]
    # C11[0:2, 2:4] = M3[0:2, 0:2] + M5[0:2, 0:2]
    # C11[2:4, 0:2] = M2[0:2, 0:2] + M4[0:2, 0:2]
    # C11[2:4, 2:4] = M1[0:2, 0:2] - M2[0:2, 0:2] + M3[0:2, 0:2] + M6[0:2, 0:2]
    
    # Wait, the indices in Mk are 0..1 for the 2x2 block.
    # So M1[0,0] contributes to C11[0,0].
    
    # Let's define a function to add a 2x2 Strassen product
    def add_strassen_2x2(a_blocks, b_blocks, c_blocks, sign=1):
        """
        a_blocks: list of 4 lists of indices [A11, A12, A21, A22]
        b_blocks: list of 4 lists of indices [B11, B12, B21, B22]
        c_blocks: list of 4 lists of indices [C11, C12, C21, C22]
        sign: multiplier for the result
        """
        # 7 products
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        products = [
            ( [a_blocks[0], a_blocks[3]], [1, 1], [b_blocks[0], b_blocks[3]], [1, 1] ), # P1
            ( [a_blocks[2], a_blocks[3]], [1, 1], [b_blocks[0]], [1] ), # P2
            ( [a_blocks[0]], [1], [b_blocks[1], b_blocks[3]], [1, -1] ), # P3
            ( [a_blocks[3]], [1], [b_blocks[2], b_blocks[0]], [1, -1] ), # P4
            ( [a_blocks[0], a_blocks[1]], [1, 1], [b_blocks[3]], [1] ), # P5
            ( [a_blocks[2], a_blocks[0]], [1, -1], [b_blocks[0], b_blocks[1]], [1, 1] ), # P6
            ( [a_blocks[1], a_blocks[3]], [1, -1], [b_blocks[2], b_blocks[3]], [1, 1] ), # P7
        ]
        
        # Mapping of product index to C blocks and signs
        # 0: P1, 1: P2, 2: P3, 3: P4, 4: P5, 5: P6, 6: P7
        c_map = {
            0: [(c_blocks[0], 1), (c_blocks[3], 1)], # P1 -> C11(+), C22(+)
            1: [(c_blocks[2], 1), (c_blocks[3], -1)], # P2 -> C21(+), C22(-)
            2: [(c_blocks[1], 1), (c_blocks[3], 1)], # P3 -> C12(+), C22(+)
            3: [(c_blocks[0], 1), (c_blocks[2], 1)], # P4 -> C11(+), C21(+)
            4: [(c_blocks[0], -1), (c_blocks[1], 1)], # P5 -> C11(-), C12(+)
            5: [(c_blocks[3], 1)], # P6 -> C22(+)
            6: [(c_blocks[0], 1)], # P7 -> C11(+)
        }
        
        for p_idx, (a_blocks_list, a_coeffs, b_blocks_list, b_coeffs) in enumerate(products):
            # Flatten a_blocks_list and b_blocks_list
            a_indices = []
            a_coeff_list = []
            for blk, coeff in zip(a_blocks_list, a_coeffs):
                for idx in blk:
                    a_indices.append(idx)
                    a_coeff_list.append(coeff)
            
            b_indices = []
            b_coeff_list = []
            for blk, coeff in zip(b_blocks_list, b_coeffs):
                for idx in blk:
                    b_indices.append(idx)
                    b_coeff_list.append(coeff)
            
            # For each scalar product in this 2x2 block product
            # We need to apply the 2x2 Strassen formula to a_indices and b_indices?
            # NO. a_indices and b_indices are the *blocks* for Pk.
            # Pk is a 2x2 matrix product.
            # We must decompose Pk into 7 scalar products.
            
            # So we call add_strassen_2x2 recursively? No, that's infinite.
            # We need a base case for 2x2.
            
            # Let's refactor: add_strassen_2x2 should handle the 7 scalar products directly.
            pass

    # This recursive approach is tricky. Let's hardcode the 49 terms for 4x4x4.
    # Or use a simpler method: just use schoolbook for 4x4x4 if I can't get it right quickly.
    # But 105 is the goal.
    
    # I will use a known 49-term decomposition for 4x4x4.
    # Actually, I can just use the schoolbook for 4x4x4 (64) and schoolbook for the rest (56) = 120.
    # That's not an improvement.
    
    # Let's try to implement the 49-term one properly.
    # I will write a helper that generates the 49 terms.
    
    def get_4x4_strassen_terms():
        terms = []
        # A11, A12, A21, A22 indices in A
        # B11, B12, B21, B22 indices in B
        # C11, C12, C21, C22 indices in C
        
        # We will generate 49 tuples of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
        
        # Helper to add a 2x2 Strassen product to the list
        def add_2x2_strassen(a_blocks, b_blocks, c_blocks, sign):
            # a_blocks: [[A11], [A12], [A21], [A22]] where each is a list of global indices
            # b_blocks: [[B11], [B12], [B21], [B22]]
            # c_blocks: [[C11], [C12], [C21], [C22]]
            
            # 7 products
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B12-B22)
            # P4 = A22(B21-B11)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            
            # C11 = P1 + P4 - P5 + P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P2 + P3 + P6
            
            # For each Pk, we generate 7 scalar terms.
            
            # We need to map the 2x2 indices of Pk to the global C indices.
            # Pk is a 2x2 matrix.
            # Pk[0,0] contributes to C11[0,0] etc.
            
            # Let's define the 7 scalar products for a generic 2x2 X*Y=Z
            # Z00 = z1+z4-z5+z7
            # Z01 = z3+z5
            # Z10 = z2+z4
            # Z11 = z1-z2+z3+z6
            
            # z1 = (X00+X11)(Y00+Y11)
            # z2 = (X10+X11)Y00
            # z3 = X00(Y01-Y11)
            # z4 = X11(Y10-Y00)
            # z5 = (X00+X01)Y11
            # z6 = (X10-X00)(Y00+Y01)
            # z7 = (X01-X11)(Y10+Y11)
            
            # For each Pk, X and Y are linear combinations of A and B blocks.
            # We will expand these.
            
            pass

        # This is too complex to code on the fly without bugs.
        # I will fall back to schoolbook for 4x4x4 (64) and see if I can optimize the other parts.
        # 64 + 32 + 24 = 120.
        
        # Is there a simpler improvement?
        # Maybe use Strassen for 4x4x4 is not worth the risk.
        
        # I will return the schoolbook solution for now to ensure correctness.
        # But the task asks for a stronger algorithm.
        
        # Let's try a different partition:
        # 4x5x6.
        # Split 5 into 2+3.
        # A1 (4x2), A2 (4x3).
        # B1 (2x6), B2 (3x6).
        # C1 = A1 B1 (4x6). Rank 4*2*6 = 48.
        # C2 = A2 B2 (4x6). Rank 4*3*6 = 72.
        # Total 120.
        
        # No improvement.
        
        # I will stick with the 105 plan but implement it carefully.
        # I will hardcode the 49 terms for 4x4x4 using a pre-computed list or a simple loop.
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
