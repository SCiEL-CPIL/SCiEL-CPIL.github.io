"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    # Strassen 2x2 (Rank 7)
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, -1, 1, 1, -1, 0]
    ])
    V_S = np.array([
        [1, 0, 1, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, -1]
    ])
    W_S = np.array([
        [1, 0, 0, -1, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, 1, 0]
    ])

    # Laderman 3x3 (Rank 7)
    # U_L: 9x7, V_L: 9x7, W_L: 9x7
    U_L = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    V_L = np.array([
        [1, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    W_L = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])

    # Construct 6x6 decomposition (Rank 49)
    # U_6: 36x49
    # Row index: r = 3*i2 + i3, col part 3*j2 + j3. Flat: (3*i2 + i3)*6 + (3*j2 + j3)
    # Wait, A is 6x6. Row R, Col C.
    # R = 3*i2 + i3, C = 3*j2 + j3.
    # U_6[flat_idx, k] = U_S[i2, j2, k2] * U_L[i3, j3, k3]
    # k = k2*7 + k3
    
    U_6 = np.zeros((36, 49))
    V_6 = np.zeros((36, 49))
    W_6 = np.zeros((36, 49))
    
    for i2 in range(2):
        for j2 in range(2):
            for i3 in range(3):
                for j3 in range(3):
                    r_S = i2 * 2 + j2
                    r_L = i3 * 3 + j3
                    flat_idx = (3 * i2 + i3) * 6 + (3 * j2 + j3)
                    
                    for k2 in range(7):
                        for k3 in range(7):
                            k = k2 * 7 + k3
                            U_6[flat_idx, k] = U_S[r_S, k2] * U_L[r_L, k3]
                            V_6[flat_idx, k] = V_S[r_S, k2] * V_L[r_L, k3]
                            W_6[flat_idx, k] = W_S[r_S, k2] * W_L[r_L, k3]

    # Restrict to 5x5
    # Indices 0..4 correspond to:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (0,2)
    # 3: (1,0)
    # 4: (1,1)
    # 5: (1,2) is excluded.
    
    # We need to extract rows 0..4 from U_6, V_6, W_6.
    # But U_6 rows are flat indices 0..35.
    # The valid indices for 5x5 are those where the 6th index (1,2) is not involved.
    # In our mapping R = 3*i2 + i3:
    # i2=0, i3=0 -> R=0
    # i2=0, i3=1 -> R=1
    # i2=0, i3=2 -> R=2
    # i2=1, i3=0 -> R=3
    # i2=1, i3=1 -> R=4
    # i2=1, i3=2 -> R=5 (Excluded)
    
    # So we keep flat indices where R < 5 and C < 5?
    # No, the flat index is for the entire 6x6 matrix.
    # We just keep the rows corresponding to the 25 elements of the 5x5 submatrix.
    # The 5x5 submatrix is the top-left 5x5 of the 6x6.
    # Rows 0-4, Cols 0-4.
    # Flat indices:
    # R=0: C=0,1,2,3,4 -> 0,1,2,3,4
    # R=1: C=0,1,2,3,4 -> 6,7,8,9,10
    # R=2: C=0,1,2,3,4 -> 12,13,14,15,16
    # R=3: C=0,1,2,3,4 -> 18,19,20,21,22
    # R=4: C=0,1,2,3,4 -> 24,25,26,27,28
    
    valid_rows = []
    for R in range(5):
        for C in range(5):
            valid_rows.append(R * 6 + C)
            
    U = U_6[valid_rows, :]
    V = V_6[valid_rows, :]
    W = W_6[valid_rows, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
