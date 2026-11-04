"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56 using Strassen on 2x2 blocks
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 8 blocks: 2 row-blocks (b) x 4 col-blocks (c)
    for b in range(2):
        for c in range(4):
            # Local indices for A (2x2), B (2x2), C (2x2)
            # A rows: 0, 1. A cols: 2*b, 2*b+1
            # B rows: 2*b, 2*b+1. B cols: 2*c, 2*c+1
            # C rows: 0, 1. C cols: 2*c, 2*c+1
            
            # Strassen Multiplications
            # M1 = (A11 + A22) * (B11 + B22)
            # A11 is A[0, 2b], A22 is A[1, 2b+1]
            # B11 is B[2b, 2c], B22 is B[2b+1, 2c+1]
            
            # Helper to add to U, V, W
            def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
                nonlocal col
                U[u_idx][col] += u_val
                V[v_idx][col] += v_val
                W[w_idx][col] += w_val
                col += 1

            # Indices
            a_11 = 0 * m + 2 * b
            a_12 = 0 * m + 2 * b + 1
            a_21 = 1 * m + 2 * b
            a_22 = 1 * m + 2 * b + 1
            
            b_11 = 2 * b * p + 2 * c
            b_12 = 2 * b * p + 2 * c + 1
            b_21 = (2 * b + 1) * p + 2 * c
            b_22 = (2 * b + 1) * p + 2 * c + 1
            
            c_11 = 2 * c * n + 0
            c_12 = 2 * c * n + 1
            c_21 = (2 * c + 1) * n + 0
            c_22 = (2 * c + 1) * n + 1

            # M1 = (A11 + A22) * (B11 + B22)
            add_term(a_11, b_11, c_11, 1, 1, 1)
            add_term(a_11, b_22, c_11, 1, 1, 1)
            add_term(a_22, b_11, c_11, 1, 1, 1)
            add_term(a_22, b_22, c_11, 1, 1, 1)
            
            # Wait, this is not how Strassen works in tensor decomposition.
            # Strassen forms linear combinations FIRST, then multiplies.
            # M1 = L_A1 * L_B1
            # L_A1 = A11 + A22
            # L_B1 = B11 + B22
            
            # We need to reset col to 0 and do this properly.
            pass

    # Re-implementing properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for b in range(2):
        for c in range(4):
            a_11 = 0 * m + 2 * b
            a_12 = 0 * m + 2 * b + 1
            a_21 = 1 * m + 2 * b
            a_22 = 1 * m + 2 * b + 1
            
            b_11 = 2 * b * p + 2 * c
            b_12 = 2 * b * p + 2 * c + 1
            b_21 = (2 * b + 1) * p + 2 * c
            b_22 = (2 * b + 1) * p + 2 * c + 1
            
            c_11 = 2 * c * n + 0
            c_12 = 2 * c * n + 1
            c_21 = (2 * c + 1) * n + 0
            c_22 = (2 * c + 1) * n + 1

            # M1 = (A11 + A22) * (B11 + B22)
            # C11 += M1, C22 += M1
            # U: A11, A22. V: B11, B22.
            # We add 1 to U[A11], U[A22], V[B11], V[B22].
            # W[C11] += 1, W[C22] += 1.
            
            U[a_11][col] += 1
            U[a_22][col] += 1
            V[b_11][col] += 1
            V[b_22][col] += 1
            W[c_11][col] += 1
            W[c_22][col] += 1
            col += 1

            # M2 = (A21 + A22) * B11
            # C21 += M2, C22 -= M2
            U[a_21][col] += 1
            U[a_22][col] += 1
            V[b_11][col] += 1
            W[c_21][col] += 1
            W[c_22][col] -= 1
            col += 1

            # M3 = A11 * (B21 - B22)
            # C12 += M3, C22 += M3
            U[a_11][col] += 1
            V[b_21][col] += 1
            V[b_22][col] -= 1
            W[c_12][col] += 1
            W[c_22][col] += 1
            col += 1

            # M4 = A22 * (B12 - B11)
            # C11 += M4, C21 += M4
            U[a_22][col] += 1
            V[b_12][col] += 1
            V[b_11][col] -= 1
            W[c_11][col] += 1
            W[c_21][col] += 1
            col += 1

            # M5 = (A11 + A12) * B22
            # C11 -= M5, C12 += M5
            U[a_11][col] += 1
            U[a_12][col] += 1
            V[b_22][col] += 1
            W[c_11][col] -= 1
            W[c_12][col] += 1
            col += 1

            # M6 = (A21 - A11) * (B11 + B12)
            # C22 += M6
            U[a_21][col] += 1
            U[a_11][col] -= 1
            V[b_11][col] += 1
            V[b_12][col] += 1
            W[c_22][col] += 1
            col += 1

            # M7 = (A12 - A22) * (B21 + B22)
            # C11 += M7, C22 += M7
            U[a_12][col] += 1
            U[a_22][col] -= 1
            V[b_21][col] += 1
            V[b_22][col] += 1
            W[c_11][col] += 1
            W[c_22][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
