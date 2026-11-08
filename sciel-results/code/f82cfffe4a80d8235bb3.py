import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add terms
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        U[u_idx, col] = coeff
        V[v_idx, col] = 1.0
        W[w_idx, col] = 1.0
        col += 1

    # --- Sub-problem 1: T_{2,3,2} for B cols 0,1 ---
    # A indices:
    # A_L (col 0): 0 (A00), 3 (A10)
    # A_R (cols 1,2): 1 (A01), 2 (A02), 4 (A11), 5 (A12)
    # B1 indices (global):
    # B_T (row 0): 0 (B00), 1 (B01)
    # B_B (rows 1,2): 7 (B10), 8 (B11), 14 (B20), 15 (B21)
    # C1 indices (global):
    # C00: 0, C01: 1, C10: 7, C11: 8
    
    # Part A: A_L * B_T (Rank 4)
    # C00 += A00 * B00
    add_term(0, 0, 0)
    # C01 += A00 * B01
    add_term(0, 1, 1)
    # C10 += A10 * B00
    add_term(3, 0, 7)
    # C11 += A10 * B01
    add_term(3, 1, 8)
    
    # Part B: A_R * B_B (Rank 7, Strassen)
    # A_R: A01(1), A02(2), A11(4), A12(5)
    # B_B: B10(7), B11(8), B20(14), B21(15)
    
    # M1 = (A01 + A12) * (B10 + B20)
    # U: 1, 5. V: 7, 14. W: 0, 1, 7, 8 (with coeffs)
    # We need to decompose this into rank-1 terms.
    # Strassen adds 7 multiplications.
    # We will manually expand the Strassen terms into the U,V,W matrices.
    
    # M1
    # (A01 + A12)(B10 + B20) = A01B10 + A01B20 + A12B10 + A12B20
    # This is not a single rank-1 term. Strassen combines sums in U and V.
    # U_vector for M1: [0, 0, 0, 0, 1, 1] (indices 1, 5 active)
    # V_vector for M1: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1] (indices 7, 14 active)
    # W_vector for M1:
    # C00 += M1, C01 += 0, C10 += 0, C11 -= M1
    # W indices: 0 (coeff 1), 8 (coeff -1)
    
    def add_strassen_term(u_vec, v_vec, w_vec):
        nonlocal col
        # u_vec, v_vec, w_vec are dicts or arrays of coefficients
        # We just set the non-zero entries
        for idx, val in u_vec.items():
            if val != 0:
                U[idx, col] += val
        for idx, val in v_vec.items():
            if val != 0:
                V[idx, col] += val
        for idx, val in w_vec.items():
            if val != 0:
                W[idx, col] += val
        col += 1

    # M1
    add_strassen_term({1: 1, 5: 1}, {7: 1, 14: 1}, {0: 1, 8: -1})
    
    # M2 = (A11 + A12) * B10
    add_strassen_term({4: 1, 5: 1}, {7: 1}, {7: 1, 8: -1})
    
    # M3 = A01 * (B11 - B21)
    add_strassen_term({1: 1}, {8: 1, 15: -1}, {1: 1, 8: 1})
    
    # M4 = A12 * (B20 - B10)
    add_strassen_term({5: 1}, {14: 1, 7: -1}, {0: 1, 7: 1})
    
    # M5 = (A01 + A02) * B20
    add_strassen_term({1: 1, 2: 1}, {14: 1}, {0: -1, 1: 1})
    
    # M6 = (A11 - A01) * (B10 + B11)
    add_strassen_term({4: 1, 1: -1}, {7: 1, 8: 1}, {8: 1})
    
    # M7 = (A02 - A12) * (B20 + B21)
    add_strassen_term({2: 1, 5: -1}, {14: 1, 15: 1}, {0: 1, 8: 1})
    
    # --- Sub-problem 2: T_{2,3,2} for B cols 2,3 ---
    # B2 indices (global):
    # B_T (row 0): 2 (B02), 3 (B03)
    # B_B (rows 1,2): 9 (B12), 10 (B13), 16 (B22), 17 (B23)
    # C2 indices (global):
    # C02: 2, C03: 3, C12: 9, C13: 10
    
    # Part A: A_L * B_T (Rank 4)
    add_term(0, 2, 2)   # C02 += A00 * B02
    add_term(0, 3, 3)   # C03 += A00 * B03
    add_term(3, 2, 9)   # C12 += A10 * B02
    add_term(3, 3, 10)  # C13 += A10 * B03
    
    # Part B: A_R * B_B (Rank 7, Strassen)
    # A_R same: 1, 2, 4, 5
    # B_B: 9, 10, 16, 17
    
    # M1
    add_strassen_term({1: 1, 5: 1}, {9: 1, 16: 1}, {2: 1, 10: -1})
    # M2
    add_strassen_term({4: 1, 5: 1}, {9: 1}, {9: 1, 10: -1})
    # M3
    add_strassen_term({1: 1}, {10: 1, 17: -1}, {3: 1, 10: 1})
    # M4
    add_strassen_term({5: 1}, {16: 1, 9: -1}, {2: 1, 9: 1})
    # M5
    add_strassen_term({1: 1, 2: 1}, {16: 1}, {2: -1, 3: 1})
    # M6
    add_strassen_term({4: 1, 1: -1}, {9: 1, 10: 1}, {10: 1})
    # M7
    add_strassen_term({2: 1, 5: -1}, {16: 1, 17: 1}, {2: 1, 10: 1})
    
    # --- Sub-problem 3: T_{2,3,3} for B cols 4,5,6 ---
    # B3 indices (global):
    # B_T (row 0): 4 (B04), 5 (B05), 6 (B06)
    # B_B (rows 1,2): 11 (B14), 12 (B15), 13 (B16), 18 (B24), 19 (B25), 20 (B26)
    # C3 indices (global):
    # C04: 4, C05: 5, C06: 6
    # C14: 11, C15: 12, C16: 13
    
    # Part A: A_L * B_T (Rank 6)
    # C04 += A00 * B04
    add_term(0, 4, 4)
    # C05 += A00 * B05
    add_term(0, 5, 5)
    # C06 += A00 * B06
    add_term(0, 6, 6)
    # C14 += A10 * B04
    add_term(3, 4, 11)
    # C15 += A10 * B05
    add_term(3, 5, 12)
    # C16 += A10 * B06
    add_term(3, 6, 13)
    
    # Part B: A_R * B_B (Rank 11)
    # A_R: 1, 2, 4, 5
    # B_B: 11, 12, 13 (row 1), 18, 19, 20 (row 2)
    # This is T_{2,2,3}.
    # Split B_B into B_B1 (2x2) and B_B2 (2x1).
    # B_B1 cols 0,1 (global 11,12 and 18,19).
    # B_B2 col 2 (global 13 and 20).
    
    # Sub-part B1: T_{2,2,2} (Strassen)
    # B_B1:
    # Row 1 (local row 0): 11, 12
    # Row 2 (local row 1): 18, 19
    # C_B1:
    # Row 0 (local row 0): 4, 5
    # Row 1 (local row 1): 11, 12
    
    # M1
    add_strassen_term({1: 1, 5: 1}, {11: 1, 18: 1}, {4: 1, 12: -1})
    # M2
    add_strassen_term({4: 1, 5: 1}, {11: 1}, {11: 1, 12: -1})
    # M3
    add_strassen_term({1: 1}, {12: 1, 19: -1}, {5: 1, 12: 1})
    # M4
    add_strassen_term({5: 1}, {18: 1, 11: -1}, {4: 1, 11: 1})
    # M5
    add_strassen_term({1: 1, 2: 1}, {18: 1}, {4: -1, 5: 1})
    # M6
    add_strassen_term({4: 1, 1: -1}, {11: 1, 12: 1}, {12: 1})
    # M7
    add_strassen_term({2: 1, 5: -1}, {18: 1, 19: 1}, {4: 1, 12: 1})
    
    # Sub-part B2: T_{2,2,1} (Rank 4)
    # B_B2:
    # Row 1: 13
    # Row 2: 20
    # C_B2:
    # Row 0: 6
    # Row 1: 13
    
    # C06 += A01 * B13 + A02 * B23 (local indices)
    # Global: C06 (6) += A01 (1) * B14 (13) + A02 (2) * B24 (20)
    add_term(1, 13, 6)
    add_term(2, 20, 6)
    
    # C16 += A11 * B13 + A12 * B23
    # Global: C16 (13) += A11 (4) * B14 (13) + A12 (5) * B24 (20)
    add_term(4, 13, 13)
    add_term(5, 20, 13)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
