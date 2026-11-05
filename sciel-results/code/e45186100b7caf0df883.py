"""
Program to construct a rank-minimal exact tensor decomposition for matrix multiplication.
Dimensions: 4x5 * 5x6.

Approach: Use a block-partitioned decomposition where:
- The 4x4 block of A (rows 0-3, cols 0-3) and corresponding 4x4 block of B (rows 0-3, cols 0-3)
  contribute to C[0:4, 0:4] via a 4x4x4 Strassen decomposition (rank 49).
- All other contributions are handled by schoolbook (rank 120 - 49 + 49 = 120? No, let's compute carefully).

Actually, let's think about this more carefully. The full schoolbook is 4*5*6 = 120.

Let's partition:
A is 4x5, B is 5x6, C is 4x6.

We can split A into A1 (4x4) and A2 (4x1), where A1 = A[:, 0:4], A2 = A[:, 4:5].
We can split B into B1 (5x4) and B2 (5x2), where B1 = B[0:4, :], B2 = B[4:5, :].
Wait, that doesn't work cleanly because B is 5x6.

Let me reconsider. Split by the middle dimension m=5:
A = [A_left (4x4) | A_right (4x1)]
B = [B_top (4x6); B_bottom (1x6)]

Then C = A_left * B_top + A_right * B_bottom.

A_left * B_top is 4x4 * 4x6 = 4x6, rank 4*4*6 = 96 by schoolbook.
A_right * B_bottom is 4x1 * 1x6 = 4x6, rank 4*1*6 = 24 by schoolbook.
Total: 96 + 24 = 120. Same as before.

To improve, apply Strassen to the 4x4x4 part of A_left * B_top.
A_left is 4x4, B_top is 4x6. Split B_top into B_top1 (4x4) and B_top2 (4x2).
C_left = A_left * B_top1 + A_left * B_top2.

A_left * B_top1 is 4x4 * 4x4, which can be done with Strassen at rank 49.
A_left * B_top2 is 4x4 * 4x2, schoolbook rank 4*4*2 = 32.
A_right * B_bottom is 4x1 * 1x6, schoolbook rank 4*1*6 = 24.

Total: 49 + 32 + 24 = 105.

This gives rank 105, which is less than 120.
"""

import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    # We'll build the decomposition in three parts:
    # Part 1: Strassen for A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # Part 2: Schoolbook for A[0:4,0:4] * B[0:4,2:6] -> C[0:4,2:6] (but only the part not in Part 1)
    #         Actually, B[0:4,2:6] is 4x2, so A[0:4,0:4] * B[0:4,2:6] is 4x4 * 4x2
    # Part 3: Schoolbook for A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6]
    
    # Let's be more precise with the indexing.
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x6: rows 0-4, cols 0-5
    # C is 4x6: rows 0-3, cols 0-5
    
    # Part 1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    #   A1 = A[:, 0:4] (4x4), B1 = B[0:4, 0:4] (4x4)
    #   Uses Strassen, rank 49
    #   
    # Part 2: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
    #   A1 = A[:, 0:4] (4x4), B2 = B[0:4, 4:6] (4x2)
    #   Schoolbook, rank 4*4*2 = 32
    #
    # Part 3: A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6]
    #   A2 = A[:, 4:5] (4x1), B3 = B[4:5, 0:6] (1x6)
    #   Schoolbook, rank 4*1*6 = 24
    #
    # Total rank: 49 + 32 + 24 = 105
    
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Strassen's algorithm for 2x2x2 has 7 multiplications.
    # For 4x4x4, we recursively apply Strassen, giving 7*7 = 49 multiplications.
    
    def strassen_2x2_terms():
        """
        Yields the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
        where indices are local 0-1 for the 2x2 blocks.
        """
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        #
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # c00 = M1 - M4 + M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M7
        
        terms = []
        
        # M1 = (a00 + a11) * (b00 + b11)
        u_idx = [(0,0), (1,1)]
        u_cof = [1, 1]
        v_idx = [(0,0), (1,1)]
        v_cof = [1, 1]
        w_idx = [(0,0), (1,1)]
        w_cof = [1, 1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M2 = (a10 + a11) * b00
        u_idx = [(1,0), (1,1)]
        u_cof = [1, 1]
        v_idx = [(0,0)]
        v_cof = [1]
        w_idx = [(1,0)]
        w_cof = [1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M3 = a00 * (b01 - b11)
        u_idx = [(0,0)]
        u_cof = [1]
        v_idx = [(0,1), (1,1)]
        v_cof = [1, -1]
        w_idx = [(0,1)]
        w_cof = [1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M4 = a11 * (b10 - b00)
        u_idx = [(1,1)]
        u_cof = [1]
        v_idx = [(1,0), (0,0)]
        v_cof = [1, -1]
        w_idx = [(0,0), (1,0)]
        w_cof = [-1, 1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M5 = (a00 + a01) * b11
        u_idx = [(0,0), (0,1)]
        u_cof = [1, 1]
        v_idx = [(1,1)]
        v_cof = [1]
        w_idx = [(0,0), (0,1)]
        w_cof = [1, 1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u_idx = [(1,0), (0,0)]
        u_cof = [1, -1]
        v_idx = [(0,0), (0,1)]
        v_cof = [1, 1]
        w_idx = [(1,0), (1,1)]
        w_cof = [1, 1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u_idx = [(0,1), (1,1)]
        u_cof = [1, -1]
        v_idx = [(1,0), (1,1)]
        v_cof = [1, 1]
        w_idx = [(0,0), (1,1)]
        w_cof = [-1, 1]
        terms.append((u_idx, u_cof, v_idx, v_cof, w_idx, w_cof))
        
        return terms
    
    def strassen_4x4_terms():
        """
        Yields the 49 Strassen terms for 4x4x4 matrix multiplication.
        Each term is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
        where indices are local (row, col) in 0-3 range.
        """
        base_terms = strassen_2x2_terms()
        result = []
        
        for outer_idx in range(7):
            ou_idx, ou_cof, ov_idx, ov_cof, ow_idx, ow_cof = base_terms[outer_idx]
            
            for inner_idx in range(7):
                iu_idx, iu_cof, iv_idx, iv_cof, iw_idx, iw_cof = base_terms[inner_idx]
                
                # Combine the outer and inner terms
                # The outer term operates on 2x2 blocks of the 4x4 matrix.
                # Each element of the outer U/V/W is itself a 2x2 matrix,
                # which is further decomposed by the inner Strassen terms.
                
                new_u_idx = []
                new_u_cof = []
                for (orow, ocol), oc in zip(ou_idx, ou_cof):
                    for (ir, ic), ic_c in zip(iu_idx, iu_cof):
                        # Map outer block (orow, ocol) with inner (ir, ic) to global 4x4
                        grow = orow * 2 + ir
                        gcol = ocol * 2 + ic
                        new_u_idx.append((grow, gcol))
                        new_u_cof.append(oc * ic_c)
                
                new_v_idx = []
                new_v_cof = []
                for (orow, ocol), oc in zip(ov_idx, ov_cof):
                    for (ir, ic), ic_c in zip(iv_idx, iv_cof):
                        grow = orow * 2 + ir
                        gcol = ocol * 2 + ic
                        new_v_idx.append((grow, gcol))
                        new_v_cof.append(oc * ic_c)
                
                new_w_idx = []
                new_w_cof = []
                for (orow, ocol), oc in zip(ow_idx, ow_cof):
                    for (ir, ic), ic_c in zip(iw_idx, iw_cof):
                        grow = orow * 2 + ir
                        gcol = ocol * 2 + ic
                        new_w_idx.append((grow, gcol))
                        new_w_cof.append(oc * ic_c)
                
                result.append((new_u_idx, new_u_cof, new_v_idx, new_v_cof, new_w_idx, new_w_cof))
        
        return result
    
    strassen_terms = strassen_4x4_terms()
    
    for (u_idx, u_cof, v_idx, v_cof, w_idx, w_cof) in strassen_terms:
        # A1 is A[:, 0:4], so row i in A1 corresponds to row i in A, col j in A1 corresponds to col j in A
        # B1 is B[0:4, 0:4], so row j in B1 corresponds to row j in B, col k in B1 corresponds to col k in B
        # C1 is C[0:4, 0:4], so row i in C1 corresponds to row i in C, col k in C1 corresponds to col k in C
        
        for (ui, uj), uc in zip(u_idx, u_cof):
            # A1[ui, uj] -> A[ui, uj] since A1 = A[:, 0:4]
            U[ui * m + uj][col] = uc
        
        for (vj, vk), vc in zip(v_idx, v_cof):
            # B1[vj, vk] -> B[vj, vk] since B1 = B[0:4, 0:4]
            V[vj * p + vk][col] = vc
        
        for (wk, wi), wc in zip(w_idx, w_cof):
            # C1[wi, wk] -> C[wi, wk] since C1 = C[0:4, 0:4]
            # W index: k*n + i, so W[wk * n + wi]
            W[wk * n + wi][col] = wc
        
        col += 1
    
    # --- Part 2: Schoolbook for A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6] ---
    # A1 = A[:, 0:4] (4x4), B2 = B[0:4, 4:6] (4x2)
    # C2 = C[0:4, 4:6] (4x2)
    # For each i in 0-3, j in 0-3, k in 4-5:
    #   U[i*m + j, col] = 1
    #   V[j*p + k, col] = 1
    #   W[k*n + i, col] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # --- Part 3: Schoolbook for A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6] ---
    # A2 = A[:, 4:5] (4x1), B3 = B[4:5, 0:6] (1x6)
    # C3 = C[0:4, 0:6] (4x6)
    # For each i in 0-3, j=4, k in 0-5:
    #   U[i*m + 4, col] = 1
    #   V[4*p + k, col] = 1
    #   W[k*n + i, col] = 1
    
    for i in range(4):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
