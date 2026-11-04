"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M0 = (x00+x11)(y00+y11)
    # M1 = (x10+x11)y00
    # M2 = x00(y01-y11)
    # M3 = x11(y10-y00)
    # M4 = (x00+x01)y11
    # M5 = (x10-x00)(y00+y10)
    # M6 = (x01-x11)(y11+y10)
    
    U_local = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 0], [0, 1]],  # M3
        [[1, 1], [0, 0]],  # M4
        [[-1, 0], [1, 0]], # M5
        [[0, 1], [0, -1]]  # M6
    ]
    
    V_local = [
        [[1, 0], [0, 1]],  # M0
        [[1, 0], [0, 0]],  # M1
        [[0, 1], [0, -1]], # M2
        [[-1, 0], [1, 0]], # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, 1]]   # M6
    ]
    
    W_local = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, -1]], # M1
        [[0, 1], [0, 1]],  # M2
        [[1, 0], [1, 0]],  # M3
        [[-1, 1], [0, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 0], [0, 0]]   # M6
    ]
    
    col = 0
    for k in range(3):
        for s in range(4):
            if s == 0:
                i_off_x, j_off_x = 0, 0
                j_off_y, k_off_y = 0, 2*k
                k_off_z, i_off_z = 2*k, 0
            elif s == 1:
                i_off_x, j_off_x = 0, 2
                j_off_y, k_off_y = 2, 2*k
                k_off_z, i_off_z = 2*k, 0
            elif s == 2:
                i_off_x, j_off_x = 2, 0
                j_off_y, k_off_y = 0, 2*k
                k_off_z, i_off_z = 2*k, 2
            elif s == 3:
                i_off_x, j_off_x = 2, 2
                j_off_y, k_off_y = 2, 2*k
                k_off_z, i_off_z = 2*k, 2
            
            for t in range(7):
                for a in range(2):
                    for b in range(2):
                        val = U_local[t][a][b]
                        if val != 0:
                            i = i_off_x + a
                            j = j_off_x + b
                            U[i*m + j][col] = val
                
                for c in range(2):
                    for d in range(2):
                        val = V_local[t][c][d]
                        if val != 0:
                            j = j_off_y + c
                            k = k_off_y + d
                            V[j*p + k][col] = val
                
                for e in range(2):
                    for f in range(2):
                        val = W_local[t][e][f]
                        if val != 0:
                            k = k_off_z + f
                            i = i_off_z + e
                            W[k*n + i][col] = val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
