"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 using Strassen on 2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # U, V, W contributions for each Strassen multiplier
    # X is 2x2 (indices 0,1 for rows/cols)
    # Y is 2x2
    # Z is 2x2
    
    # We need to map local 2x2 indices to global indices.
    # For A1 (cols 0,1 of A), local (i, j) -> global (i, j)
    # For A2 (cols 2,3 of A), local (i, j) -> global (i, j+2)
    
    # For B_j_top (rows 0,1 of B, cols 2j, 2j+1), local (j, k) -> global (j, 2j+k)
    # For B_j_bottom (rows 2,3 of B, cols 2j, 2j+1), local (j, k) -> global (j+2, 2j+k)
    
    # For C_j (cols 2j, 2j+1 of C), local (k, i) -> global (2j+k, i)
    # Note: W index is k*n+i. Global W index is (2j+k)*n + i.
    
    strassen_U = [
        [[1, 0], [0, 1]], # M1: X00, X11
        [[0, 0], [1, 1]], # M2: X10, X11
        [[1, 0], [0, 0]], # M3: X00
        [[0, 0], [0, 1]], # M4: X11
        [[1, 1], [0, 0]], # M5: X00, X01
        [[-1, 0], [1, 0]],# M6: X10, X00
        [[0, 1], [0, -1]] # M7: X01, X11
    ]
    strassen_V = [
        [[1, 0], [0, 1]], # M1: Y00, Y11
        [[1, 0], [0, 0]], # M2: Y00
        [[0, -1], [0, 1]],# M3: Y01, Y11
        [[0, -1], [1, 0]],# M4: Y10, Y00
        [[0, 0], [0, 1]], # M5: Y11
        [[1, 1], [0, 0]], # M6: Y00, Y01
        [[0, 1], [1, 1]]  # M7: Y10, Y11
    ]
    strassen_W = [
        [[1, 0], [0, 1]], # M1: Z00, Z11
        [[0, 0], [1, -1]],# M2: Z10, Z11
        [[0, 1], [0, 1]], # M3: Z01, Z11
        [[1, 0], [1, 0]], # M4: Z00, Z10
        [[-1, 0], [0, 1]],# M5: Z00, Z01
        [[0, 0], [0, 1]], # M6: Z11
        [[1, 0], [0, 0]]  # M7: Z00
    ]
    
    # Note: Z indices in strassen_W are (row, col) of Z.
    # Z00 -> (0,0), Z01 -> (0,1), Z10 -> (1,0), Z11 -> (1,1).
    # In our mapping, Z is C_j.
    # C_j[i, k] corresponds to global C[i, 2j+k].
    # W index is k*n + i.
    # So local Z[i, k] maps to global W index (2j+k)*n + i.
    
    for j in range(3):
        # A1 block
        for s in range(7):
            col = j * 14 + s
            # U: A1 (cols 0,1)
            for i in range(2):
                for u_j in range(2):
                    val = strassen_U[s][i][u_j]
                    if val != 0:
                        U[i * m + u_j][col] = val
            # V: B_j_top (rows 0,1, cols 2j, 2j+1)
            for v_j in range(2):
                for v_k in range(2):
                    val = strassen_V[s][v_j][v_k]
                    if val != 0:
                        V[v_j * p + (2*j + v_k)][col] = val
            # W: C_j (cols 2j, 2j+1)
            for w_i in range(2):
                for w_k in range(2):
                    val = strassen_W[s][w_i][w_k]
                    if val != 0:
                        W[(2*j + w_k) * n + w_i][col] = val
        
        # A2 block
        for s in range(7):
            col = j * 14 + 7 + s
            # U: A2 (cols 2,3)
            for i in range(2):
                for u_j in range(2):
                    val = strassen_U[s][i][u_j]
                    if val != 0:
                        U[i * m + (2 + u_j)][col] = val
            # V: B_j_bottom (rows 2,3, cols 2j, 2j+1)
            for v_j in range(2):
                for v_k in range(2):
                    val = strassen_V[s][v_j][v_k]
                    if val != 0:
                        V[(2 + v_j) * p + (2*j + v_k)][col] = val
            # W: C_j (cols 2j, 2j+1)
            for w_i in range(2):
                for w_k in range(2):
                    val = strassen_W[s][w_i][w_k]
                    if val != 0:
                        W[(2*j + w_k) * n + w_i][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
