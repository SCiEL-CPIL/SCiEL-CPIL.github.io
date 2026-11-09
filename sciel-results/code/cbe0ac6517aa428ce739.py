"""Matrix multiplication 3x4x6 - rank 66 decomposition via 3x2x2 blocks."""

def construct():
    n, m, p = 3, 4, 6
    
    # Total rank = 2 (A splits) * 3 (B splits) * 11 (rank of 3x2x2) = 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Loop over A splits (0, 1) and B splits (0, 1, 2)
    for a_split in range(2):
        for b_split in range(3):
            # Determine global indices for this 3x2x2 block
            # A: rows 0,1,2; cols 2*a_split, 2*a_split+1
            # B: rows 2*a_split, 2*a_split+1; cols 2*b_split, 2*b_split+1
            # C: rows 0,1,2; cols 2*b_split, 2*b_split+1
            
            a_cols = [2 * a_split, 2 * a_split + 1]
            b_rows = [2 * a_split, 2 * a_split + 1]
            b_cols = [2 * b_split, 2 * b_split + 1]
            
            # Helper to add a term
            def add_term(u_coeffs, v_coeffs, w_coeffs):
                nonlocal col
                for idx, coeff in u_coeffs:
                    U[idx][col] += coeff
                for idx, coeff in v_coeffs:
                    V[idx][col] += coeff
                for idx, coeff in w_coeffs:
                    W[idx][col] += coeff
                col += 1
            
            # Global indices for A (3x2 submatrix)
            # A[i, a_cols[j]] -> U[i*m + a_cols[j]]
            a00_idx = 0 * m + a_cols[0]
            a01_idx = 0 * m + a_cols[1]
            a10_idx = 1 * m + a_cols[0]
            a11_idx = 1 * m + a_cols[1]
            a20_idx = 2 * m + a_cols[0]
            a21_idx = 2 * m + a_cols[1]
            
            # Global indices for B (2x2 submatrix)
            # B[b_rows[i], b_cols[j]] -> V[b_rows[i]*p + b_cols[j]]
            b00_idx = b_rows[0] * p + b_cols[0]
            b01_idx = b_rows[0] * p + b_cols[1]
            b10_idx = b_rows[1] * p + b_cols[0]
            b11_idx = b_rows[1] * p + b_cols[1]
            
            # Global indices for C (3x2 submatrix)
            # C[i, b_cols[j]] -> W[b_cols[j]*n + i]
            c00_idx = b_cols[0] * n + 0
            c01_idx = b_cols[1] * n + 0
            c10_idx = b_cols[0] * n + 1
            c11_idx = b_cols[1] * n + 1
            c20_idx = b_cols[0] * n + 2
            c21_idx = b_cols[1] * n + 2
            
            # Strassen terms for top 2x2x2 (rows 0,1 of A_sub)
            
            # M1 = (a00 + a11)(b00 + b11)
            add_term(
                [(a00_idx, 1), (a11_idx, 1)],
                [(b00_idx, 1), (b11_idx, 1)],
                [(c00_idx, 1), (c11_idx, 1)]
            )
            
            # M2 = (a10 + a11)b00
            add_term(
                [(a10_idx, 1), (a11_idx, 1)],
                [(b00_idx, 1)],
                [(c10_idx, 1), (c11_idx, -1)]
            )
            
            # M3 = a00(b01 - b11)
            add_term(
                [(a00_idx, 1)],
                [(b01_idx, 1), (b11_idx, -1)],
                [(c01_idx, 1), (c11_idx, -1)]
            )
            
            # M4 = a11(b10 - b00)
            add_term(
                [(a11_idx, 1)],
                [(b10_idx, 1), (b00_idx, -1)],
                [(c00_idx, -1), (c10_idx, 1)]
            )
            
            # M5 = (a00 + a01)b11
            add_term(
                [(a00_idx, 1), (a01_idx, 1)],
                [(b11_idx, 1)],
                [(c00_idx, 1), (c01_idx, 1)]
            )
            
            # M6 = (a10 - a00)(b00 + b01)
            add_term(
                [(a10_idx, 1), (a00_idx, -1)],
                [(b00_idx, 1), (b01_idx, 1)],
                [(c11_idx, 1)]
            )
            
            # M7 = (a01 - a11)(b10 + b11)
            add_term(
                [(a01_idx, 1), (a11_idx, -1)],
                [(b10_idx, 1), (b11_idx, 1)],
                [(c11_idx, -1)]
            )
            
            # Schoolbook terms for bottom row (row 2 of A_sub)
            
            # M8 = a20 b00
            add_term(
                [(a20_idx, 1)],
                [(b00_idx, 1)],
                [(c20_idx, 1)]
            )
            
            # M9 = a21 b10
            add_term(
                [(a21_idx, 1)],
                [(b10_idx, 1)],
                [(c20_idx, 1)]
            )
            
            # M10 = a20 b01
            add_term(
                [(a20_idx, 1)],
                [(b01_idx, 1)],
                [(c21_idx, 1)]
            )
            
            # M11 = a21 b11
            add_term(
                [(a21_idx, 1)],
                [(b11_idx, 1)],
                [(c21_idx, 1)]
            )

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
