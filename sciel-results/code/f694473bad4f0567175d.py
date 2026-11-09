"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 3
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # (9, 22)
    V = [[0] * rank for _ in range(m * p)]  # (9, 22)
    W = [[0] * rank for _ in range(p * n)]  # (9, 22)
    
    def set_U(i, j, col, val):
        U[i * m + j][col] = val
    def set_V(j, k, col, val):
        V[j * p + k][col] = val
    def set_W(k, i, col, val):
        W[k * n + i][col] = val
    
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # Term 1: M2 = (A10+A11)B00 -> C10+=1, C11+=-1
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    # Term 2: M3 = A00(B01-B11) -> C00+=-1, C01+=1
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(0, 0, 2, -1); set_W(1, 0, 2, 1)
    
    # Term 3: M4 = A11(B10-B00) -> C00+=1, C10+=1
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    # Term 4: M5 = (A00+A01)B11 -> C01+=1, C11+=1
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(1, 0, 4, 1); set_W(1, 1, 4, 1)
    
    # Term 5: M6 = (A10-A00)(B00+B01) -> C10+=1
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(0, 1, 5, 1)
    
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00+=-1, C11+=1
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, -1); set_W(1, 1, 6, 1)
    
    # Terms 7-9: C[0,2] = A00*B02 + A01*B12 + A02*B22
    for j, col in enumerate([7, 8, 9]):
        set_U(0, j, col, 1)
        set_V(j, 2, col, 1)
        set_W(2, 0, col, 1)
    
    # Terms 10-12: C[1,2] = A10*B02 + A11*B12 + A12*B22
    for j, col in enumerate([10, 11, 12]):
        set_U(1, j, col, 1)
        set_V(j, 2, col, 1)
        set_W(2, 1, col, 1)
    
    # Terms 13-15: C[2,0] = A20*B00 + A21*B10 + A22*B20
    for j, col in enumerate([13, 14, 15]):
        set_U(2, j, col, 1)
        set_V(j, 0, col, 1)
        set_W(0, 2, col, 1)
    
    # Terms 16-18: C[2,1] = A20*B01 + A21*B11 + A22*B21
    for j, col in enumerate([16, 17, 18]):
        set_U(2, j, col, 1)
        set_V(j, 1, col, 1)
        set_W(1, 2, col, 1)
    
    # Terms 19-21: C[2,2] = A20*B02 + A21*B12 + A22*B22
    for j, col in enumerate([19, 20, 21]):
        set_U(2, j, col, 1)
        set_V(j, 2, col, 1)
        set_W(2, 2, col, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
