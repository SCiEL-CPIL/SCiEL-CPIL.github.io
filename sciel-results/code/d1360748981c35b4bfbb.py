"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set entries
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: P1
    set_u(0, 0, 0, 1); set_u(0, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(1, 0, 0, 1); set_w(1, 1, 0, 1)
    
    # Term 2: P2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1); set_v(1, 1, 1, 1)
    set_w(0, 0, 1, -1); set_w(1, 1, 1, 1)
    
    # Term 3: P3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 0, 2, -1)
    set_w(0, 1, 2, 1)
    
    # Term 4: P4
    set_u(1, 1, 3, 1)
    set_v(0, 0, 3, 1); set_v(0, 1, 3, -1)
    set_w(0, 0, 3, 1); set_w(1, 1, 3, -1)
    
    # Term 5: P5
    set_u(0, 0, 4, 1); set_u(1, 1, 4, 1)
    set_v(0, 0, 4, 1); set_v(1, 1, 4, 1)
    set_w(0, 0, 4, 1); set_w(0, 1, 4, 1)
    
    # Term 6: P6
    set_u(0, 1, 5, 1); set_u(1, 1, 5, 1)
    set_v(0, 1, 5, 1); set_v(1, 1, 5, -1)
    set_w(0, 0, 5, 1)
    
    # Term 7: P7
    set_u(0, 0, 6, 1); set_u(1, 0, 6, -1)
    set_v(0, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(1, 0, 6, 1); set_w(1, 1, 6, 1)
    
    # Term 8: Q1
    set_u(0, 2, 7, 1)
    set_v(2, 0, 7, 1)
    set_w(0, 0, 7, 1)
    
    # Term 9: Q2
    set_u(0, 2, 8, 1)
    set_v(2, 1, 8, 1)
    set_w(1, 0, 8, 1)
    
    # Term 10: Q3
    set_u(1, 2, 9, 1)
    set_v(2, 0, 9, 1)
    set_w(0, 1, 9, 1)
    
    # Term 11: Q4
    set_u(1, 2, 10, 1)
    set_v(2, 1, 10, 1)
    set_w(1, 1, 10, 1)
    
    # Term 12: S1
    set_u(0, 0, 11, 1)
    set_v(0, 2, 11, 1)
    set_w(2, 0, 11, 1)
    
    # Term 13: S2
    set_u(0, 1, 12, 1)
    set_v(1, 2, 12, 1)
    set_w(2, 0, 12, 1)
    
    # Term 14: S3
    set_u(1, 0, 13, 1)
    set_v(0, 2, 13, 1)
    set_w(2, 1, 13, 1)
    
    # Term 15: S4
    set_u(1, 1, 14, 1)
    set_v(1, 2, 14, 1)
    set_w(2, 1, 14, 1)
    
    # Term 16: T1
    set_u(0, 2, 15, 1)
    set_v(2, 2, 15, 1)
    set_w(2, 0, 15, 1)
    
    # Term 17: T2
    set_u(1, 2, 16, 1)
    set_v(2, 2, 16, 1)
    set_w(2, 1, 16, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
