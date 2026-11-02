import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 56 construction using 8 Strassen blocks
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for M1..M7
    # M1: X00+X11, Y00+Y11
    # M2: X10+X11, Y00
    # M3: X00, Y10-Y11
    # M4: X11, Y01-Y00
    # M5: X00+X10, Y11
    # M6: X11-X00, Y00+Y10
    # M7: X01-X11, Y11+Y01
    
    # Reconstruction:
    # Z00 = M1 + M4 - M5 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M6 + M7
    
    # We define the 8 blocks
    # Block l: A_sub (2x2) and B_sub (2x2)
    # A is 2x4. A_sub is either A[:, 0:2] (A1) or A[:, 2:4] (A2).
    # B is 4x8. B_sub is B[rows, cols].
    
    # A1 corresponds to j=0,1. A2 corresponds to j=2,3.
    # B blocks:
    # B00: rows 0,1; cols 0,1
    # B20: rows 2,3; cols 0,1
    # B01: rows 0,1; cols 2,3
    # B21: rows 2,3; cols 2,3
    # B02: rows 0,1; cols 4,5
    # B22: rows 2,3; cols 4,5
    # B03: rows 0,1; cols 6,7
    # B23: rows 2,3; cols 6,7
    
    # Mapping of local C indices (0,0), (0,1), (1,0), (1,1) to global C indices
    # Global C is 2x8. i in 0,1. k in 0..7.
    # Block l produces a 2x2 C block at specific k range.
    
    blocks = [
        # (A_type, B_rows, B_cols, C_cols)
        (0, [0, 1], [0, 1], [0, 1]),   # 1: A1 B00 -> C00
        (1, [2, 3], [0, 1], [0, 1]),   # 2: A2 B20 -> C00
        (0, [0, 1], [2, 3], [2, 3]),   # 3: A1 B01 -> C01
        (1, [2, 3], [2, 3], [2, 3]),   # 4: A2 B21 -> C01
        (0, [0, 1], [4, 5], [4, 5]),   # 5: A1 B02 -> C02
        (1, [2, 3], [4, 5], [4, 5]),   # 6: A2 B22 -> C02
        (0, [0, 1], [6, 7], [6, 7]),   # 7: A1 B03 -> C03
        (1, [2, 3], [6, 7], [6, 7]),   # 8: A2 B23 -> C03
    ]
    
    col_idx = 0
    for A_type, B_rows, B_cols, C_cols in blocks:
        # A_type 0: A[:, 0:2] -> indices 0, 1, 2, 3 in flattened A?
        # A is 2x4. Flattened index a = i*4 + j.
        # A1 (j=0,1):
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 4
        # i=1, j=1 -> 5
        # A2 (j=2,3):
        # i=0, j=2 -> 2
        # i=0, j=3 -> 3
        # i=1, j=2 -> 6
        # i=1, j=3 -> 7
        
        if A_type == 0:
            # A1 indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            # Local X00=A00, X01=A01, X10=A10, X11=A11
            a_map = {
                (0,0): 0,
                (0,1): 1,
                (1,0): 4,
                (1,1): 5
            }
        else:
            # A2 indices: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
            a_map = {
                (0,0): 2,
                (0,1): 3,
                (1,0): 6,
                (1,1): 7
            }
        
        # B is 4x8. Flattened index b = j*8 + k.
        # B_rows, B_cols are local indices 0,1 mapping to global rows/cols.
        # Local Y00=B[r0,c0], Y01=B[r0,c1], Y10=B[r1,c0], Y11=B[r1,c1]
        b_map = {
            (0,0): B_rows[0] * 8 + B_cols[0],
            (0,1): B_rows[0] * 8 + B_cols[1],
            (1,0): B_rows[1] * 8 + B_cols[0],
            (1,1): B_rows[1] * 8 + B_cols[1]
        }
        
        # C is 2x8. Flattened index c = k*2 + i.
        # Wait, the task says W[k*n+i, r].
        # So c index in W is k*n + i.
        # Local Z00=C[i=0, k_start], Z01=C[i=0, k_start+1]
        # Z10=C[i=1, k_start], Z11=C[i=1, k_start+1]
        c_start = C_cols[0]
        c_map = {
            (0,0): c_start * 2 + 0,
            (0,1): (c_start + 1) * 2 + 0,
            (1,0): c_start * 2 + 1,
            (1,1): (c_start + 1) * 2 + 1
        }
        
        # Define the 7 M terms
        # Each term is (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {local_X_index: coeff}
        # V_coeffs: dict {local_Y_index: coeff}
        # W_coeffs: dict {local_Z_index: coeff}
        
        terms = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
            # M4
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6
            ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): -1, (1,1): 1})
        ]
        
        for u_loc, v_loc, w_loc in terms:
            for (li, lj), coeff_u in u_loc.items():
                a_global = a_map[(li, lj)]
                U[a_global, col_idx] += coeff_u
            
            for (li, lj), coeff_v in v_loc.items():
                b_global = b_map[(li, lj)]
                V[b_global, col_idx] += coeff_v
            
            for (li, lj), coeff_w in w_loc.items():
                c_global = c_map[(li, lj)]
                W[c_global, col_idx] += coeff_w
            
            col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
