import time

def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    # Optimal rank for 2x4x6 matrix multiplication
    # Decomposes into two 2x2x6 blocks (split A)
    # Each 2x2x6 decomposes into three 2x2x2 blocks (split B)
    # Rank of 2x2x2 is 7 (Strassen)
    # Total Rank = 2 * 3 * 7 = 42
    rank = 42
    
    # Initialize U, V, W
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition coefficients
    # U_coeffs: list of dicts mapping (i, j) -> coeff for A (2x2)
    # V_coeffs: list of dicts mapping (j, k) -> coeff for B (2x2)
    # W_coeffs: list of dicts mapping (i, k) -> coeff for C (2x2)
    
    # Strassen formulas:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    strassen_terms = [
        # P1
        (
            {(0, 0): 1, (1, 1): 1},  # U
            {(0, 0): 1, (1, 1): 1},  # V
            {(0, 0): 1, (1, 1): 1}   # W
        ),
        # P2
        (
            {(1, 0): 1, (1, 1): 1},  # U
            {(0, 0): 1},             # V
            {(1, 0): 1, (1, 1): -1}  # W
        ),
        # P3
        (
            {(0, 0): 1},             # U
            {(1, 0): 1, (1, 1): -1}, # V
            {(0, 1): 1, (1, 1): 1}   # W
        ),
        # P4
        (
            {(1, 1): 1},             # U
            {(0, 1): 1, (0, 0): -1}, # V
            {(0, 0): 1, (1, 0): 1}   # W
        ),
        # P5
        (
            {(0, 0): 1, (0, 1): 1},  # U
            {(1, 1): 1},             # V
            {(0, 0): -1, (0, 1): 1}  # W
        ),
        # P6
        (
            {(1, 0): 1, (0, 0): -1}, # U
            {(0, 0): 1, (0, 1): 1},  # V
            {(1, 1): 1}              # W
        ),
        # P7
        (
            {(0, 1): 1, (1, 1): -1}, # U
            {(1, 0): 1, (1, 1): 1},  # V
            {(0, 0): 1}              # W
        )
    ]
    
    col_idx = 0
    # Iterate over the two halves of A (2x2 blocks)
    # A is 2x4. Halves are cols 0-1 and 2-3.
    for a_half in range(2):
        # Iterate over the three halves of B (2x2 blocks)
        # B is 4x6. 
        # We need to match the rows of B with the cols of A.
        # If a_half=0, we use A[:, 0:2]. This multiplies with B[0:2, :]?
        # Wait. C = A B.
        # C[i, k] = sum_{j=0}^{3} A[i, j] B[j, k].
        # Split sum into j=0,1 and j=2,3.
        # Term 1: sum_{j=0}^{1} A[i, j] B[j, k]. Uses A[:, 0:2] and B[0:2, :].
        # Term 2: sum_{j=2}^{3} A[i, j] B[j, k]. Uses A[:, 2:4] and B[2:4, :].
        
        # So for a_half=0, we use A indices j_loc 0,1 and B row indices j 0,1.
        # For a_half=1, we use A indices j_loc 0,1 (offset by 2) and B row indices j 0,1 (offset by 2).
        
        a_col_offset = a_half * 2
        b_row_offset = a_half * 2
        
        # Now split B[rows, cols] into 3 horizontal 2x2 blocks.
        # B is 2x6 in this sub-problem.
        # Blocks are cols 0-1, 2-3, 4-5.
        for b_block in range(3):
            b_col_offset = b_block * 2
            
            # Apply the 7 Strassen terms for this 2x2x2 sub-problem
            for u_dict, v_dict, w_dict in strassen_terms:
                # Fill U
                for (i_loc, j_loc), coeff in u_dict.items():
                    u_row = i_loc * m + (a_col_offset + j_loc)
                    U[u_row][col_idx] += coeff
                
                # Fill V
                for (j_loc, k_loc), coeff in v_dict.items():
                    # V index is j * p + k
                    # j is global row of B
                    # k is global col of B
                    v_row = (b_row_offset + j_loc) * p + (b_col_offset + k_loc)
                    V[v_row][col_idx] += coeff
                
                # Fill W
                for (i_loc, k_loc), coeff in w_dict.items():
                    # W index is k * n + i
                    # k is global col of C (same as B)
                    # i is global row of C (same as A)
                    w_row = (b_col_offset + k_loc) * n + i_loc
                    W[w_row][col_idx] += coeff
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
