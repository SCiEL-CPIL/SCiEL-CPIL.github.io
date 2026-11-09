"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen (rank 7) for 2x2 x 2x2 subblocks, schoolbook for others.
    # Partition: input rows {0,1},{2}; middle cols {0,1},{2,3},{4}; output cols {0,1},{2,3},{4,5}
    
    # Strassen rank-7 decomposition for 2x2 x 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Strassen terms: each term is (a_combo, b_combo) where a_combo is a vector over A's 4 entries, b_combo over B's 4
    # A entries indexed 0..3: (0,0),(0,1),(1,0),(1,1)
    # B entries indexed 0..3: (0,0),(0,1),(1,0),(1,1)
    strassen_terms = [
        # (a_weights, b_weights)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({1: 1, 3: 1}, {0: 1}),          # M2
        ({0: 1}, {1: 1, 0: -1}),         # M3: A11*(B21-B11) -> a: {0:1}, b: {1:1, 0:-1}
        ({3: 1}, {3: 1, 1: -1}),         # M4: A22*(B22-B12) -> a: {3:1}, b: {3:1, 1:-1}
        ({0: 1, 1: 1}, {3: 1}),          # M5
        ({1: 1, 0: -1}, {0: 1, 1: 1}),   # M6
        ({1: 1, 3: -1}, {1: 1, 3: 1}),   # M7
    ]
    # Output reconstruction: C[i,j] = sum over terms of coeff * M_t
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    strassen_output = [
        # (out_i, out_j, [(term_idx, coeff), ...])
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]
    
    U = []
    V = []
    W = []
    
    def add_term(a_row, a_col, b_row, b_col, w_row, w_col, val):
        """Add a rank-1 term: U[a_row*m+a_col] += val, V[b_row*p+b_col] += val, W[w_col*n+w_row] += val"""
        a_idx = a_row * m + a_col
        b_idx = b_row * p + b_col
        w_idx = w_col * n + w_row
        if not U:
            U.append([0])
            V.append([0])
            W.append([0])
        else:
            # Extend existing vectors
            pass
        # We'll build differently - collect all terms first
    
    # Better approach: collect all (a_idx, b_idx, w_idx, val) tuples, then build matrices
    terms = []
    
    # Define partitions
    input_blocks = [(0, 2), (2, 3)]  # row ranges for A
    middle_blocks = [(0, 2), (2, 4), (4, 5)]  # col ranges for A / row ranges for B
    output_blocks = [(0, 2), (2, 4), (4, 6)]  # col ranges for B / output col ranges
    
    for i_start, i_end in input_blocks:
        for j_start, j_end in middle_blocks:
            for k_start, k_end in output_blocks:
                i_size = i_end - i_start
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for t_idx, (a_weights, b_weights) in enumerate(strassen_terms):
                        for a_local, a_val in a_weights.items():
                            for b_local, b_val in b_weights.items():
                                a_row = i_start + a_local // 2
                                a_col = j_start + a_local % 2
                                b_row = j_start + b_local // 2
                                b_col = k_start + b_local % 2
                                val = a_val * b_val
                                terms.append((a_row, a_col, b_row, b_col, k_start, i_start, t_idx, val))
                    # Store which terms map to which outputs for this block
                    # We'll handle output reconstruction separately
                elif i_size == 1 and j_size == 1 and k_size == 1:
                    # Schoolbook: single term
                    a_row = i_start
                    a_col = j_start
                    b_row = j_start
                    b_col = k_start
                    terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    # 2x1 x 1x2: schoolbook rank 2
                    for a_r in range(2):
                        for k_c in range(2):
                            a_row = i_start + a_r
                            a_col = j_start
                            b_row = j_start
                            b_col = k_start + k_c
                            terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    # 1x2 x 2x2: schoolbook rank 4
                    for a_c in range(2):
                        for b_r in range(2):
                            for k_c in range(2):
                                a_row = i_start
                                a_col = j_start + a_c
                                b_row = j_start + b_r
                                b_col = k_start + k_c
                                terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
                elif i_size == 2 and j_size == 2 and k_size == 1:
                    # 2x2 x 2x1: schoolbook rank 4
                    for a_r in range(2):
                        for a_c in range(2):
                            b_row = j_start
                            b_col = k_start
                            terms.append((a_start + a_r, j_start + a_c, j_start, k_start, k_start, i_start, -1, 1))
                elif i_size == 1 and j_size == 2 and k_size == 1:
                    # 1x2 x 2x1: rank 2
                    for a_c in range(2):
                        b_row = j_start + a_c
                        a_row = i_start
                        a_col = j_start + a_c
                        b_col = k_start
                        terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
                elif i_size == 2 and j_size == 1 and k_size == 1:
                    # 2x1 x 1x1: rank 2
                    for a_r in range(2):
                        a_row = i_start + a_r
                        a_col = j_start
                        b_row = j_start
                        b_col = k_start
                        terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
                else:
                    # Fallback schoolbook
                    for a_r in range(i_size):
                        for a_c in range(j_size):
                            for b_r in range(j_size):
                                for k_c in range(k_size):
                                    a_row = i_start + a_r
                                    a_col = j_start + a_c
                                    b_row = j_start + b_r
                                    b_col = k_start + k_c
                                    terms.append((a_row, a_col, b_row, b_col, k_start, i_start, -1, 1))
    
    # This approach is getting complex. Let me use a cleaner method.
    # Reset and use a direct construction.
    
    # Actually, let me just return the schoolbook for now and verify the framework works.
    # Then I'll optimize.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
