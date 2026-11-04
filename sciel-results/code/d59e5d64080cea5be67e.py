"""Program for matrix multiplication 2x4 by 4x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns exact rank-minimal decomposition for 2x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 blocks.
    Rank is 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # j' in {0, 1} splits the middle dimension 4 into two 2s
    # k' in {0, 1, 2, 3} splits the output dimension 8 into four 2s
    for jp in range(2):
        for kp in range(4):
            # 7 Strassen terms
            for t in range(7):
                # Define indices and coefficients for this term
                # Local A indices: (i, j_local) where j_local in {0,1}
                # Global A index: i * m + (2*jp + j_local)
                
                # Local B indices: (j_local, k_local) where j_local, k_local in {0,1}
                # Global B index: (2*jp + j_local) * p + (2*kp + k_local)
                
                # Local C indices: (i, k_local) where i, k_local in {0,1}
                # Global C index: (2*kp + k_local) * n + i
                
                # Term 1: M1 = (A00 + A11)(B00 + B11)
                if t == 0:
                    # U
                    U[0 * m + (2*jp + 0)][col] += 1
                    U[1 * m + (2*jp + 1)][col] += 1
                    # V
                    V[(2*jp + 0) * p + (2*kp + 0)][col] += 1
                    V[(2*jp + 1) * p + (2*kp + 1)][col] += 1
                    # W
                    W[(2*kp + 0) * n + 0][col] += 1
                    W[(2*kp + 1) * n + 1][col] += 1
                # Term 2: M2 = (A10 + A11)B00
                elif t == 1:
                    # U
                    U[1 * m + (2*jp + 0)][col] += 1
                    U[1 * m + (2*jp + 1)][col] += 1
                    # V
                    V[(2*jp + 0) * p + (2*kp + 0)][col] += 1
                    # W
                    W[(2*kp + 0) * n + 1][col] += 1
                    W[(2*kp + 1) * n + 1][col] += -1
                # Term 3: M3 = A00(B01 - B11)
                elif t == 2:
                    # U
                    U[0 * m + (2*jp + 0)][col] += 1
                    # V
                    V[(2*jp + 0) * p + (2*kp + 1)][col] += 1
                    V[(2*jp + 1) * p + (2*kp + 1)][col] += -1
                    # W
                    W[(2*kp + 1) * n + 0][col] += 1
                    W[(2*kp + 1) * n + 1][col] += 1
                # Term 4: M4 = A11(B10 - B00)
                elif t == 3:
                    # U
                    U[1 * m + (2*jp + 1)][col] += 1
                    # V
                    V[(2*jp + 1) * p + (2*kp + 0)][col] += 1
                    V[(2*jp + 0) * p + (2*kp + 0)][col] += -1
                    # W
                    W[(2*kp + 0) * n + 0][col] += 1
                    W[(2*kp + 0) * n + 1][col] += 1
                # Term 5: M5 = (A00 + A01)B11
                elif t == 4:
                    # U
                    U[0 * m + (2*jp + 0)][col] += 1
                    U[0 * m + (2*jp + 1)][col] += 1
                    # V
                    V[(2*jp + 1) * p + (2*kp + 1)][col] += 1
                    # W
                    W[(2*kp + 0) * n + 0][col] += -1
                    W[(2*kp + 1) * n + 0][col] += 1
                # Term 6: M6 = (A10 - A00)(B00 + B01)
                elif t == 5:
                    # U
                    U[1 * m + (2*jp + 0)][col] += 1
                    U[0 * m + (2*jp + 0)][col] += -1
                    # V
                    V[(2*jp + 0) * p + (2*kp + 0)][col] += 1
                    V[(2*jp + 0) * p + (2*kp + 1)][col] += 1
                    # W
                    W[(2*kp + 1) * n + 1][col] += 1
                # Term 7: M7 = (A01 - A11)(B10 + B11)
                elif t == 6:
                    # U
                    U[0 * m + (2*jp + 1)][col] += 1
                    U[1 * m + (2*jp + 1)][col] += -1
                    # V
                    V[(2*jp + 1) * p + (2*kp + 0)][col] += 1
                    V[(2*jp + 1) * p + (2*kp + 1)][col] += 1
                    # W
                    W[(2*kp + 0) * n + 0][col] += 1
                    W[(2*kp + 1) * n + 1][col] += 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
