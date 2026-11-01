"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 14 (Strassen-like)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, r, c, val):
        mat[r][c] = val

    # Strassen coefficients for A (2x2)
    # Indices: 0:a00, 1:a01, 2:a10, 3:a11
    # t=0: M1 = (a00+a11)
    set_val(U, 0, 0, 1); set_val(U, 3, 0, 1)
    # t=1: M2 = (a10+a11)
    set_val(U, 2, 1, 1); set_val(U, 3, 1, 1)
    # t=2: M3 = a00
    set_val(U, 0, 2, 1)
    # t=3: M4 = a11
    set_val(U, 3, 3, 1)
    # t=4: M5 = (a00+a01)
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    # t=5: M6 = (a10-a00)
    set_val(U, 2, 5, 1); set_val(U, 0, 5, -1)
    # t=6: M7 = (a01-a11)
    set_val(U, 1, 6, 1); set_val(U, 3, 6, -1)

    # Block 1 (B cols 0,1)
    # B indices: j*p+k. j=0,1. k=0,1 -> 0,1,4,5
    # t=0: b00+b11
    set_val(V, 0, 0, 1); set_val(V, 5, 0, 1)
    # t=1: b00
    set_val(V, 0, 1, 1)
    # t=2: b01-b11
    set_val(V, 1, 2, 1); set_val(V, 5, 2, -1)
    # t=3: b10-b00
    set_val(V, 4, 3, 1); set_val(V, 0, 3, -1)
    # t=4: b11
    set_val(V, 5, 4, 1)
    # t=5: b00+b10
    set_val(V, 0, 5, 1); set_val(V, 4, 5, 1)
    # t=6: b01+b11
    set_val(V, 1, 6, 1); set_val(V, 5, 6, 1)

    # W for Block 1 (C cols 0,1)
    # C indices: k*n+i. k=0,1. i=0,1 -> 0,1,2,3
    # t=0: C00(+1), C11(+1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
    # t=1: C10(+1), C11(-1)
    set_val(W, 2, 1, 1); set_val(W, 3, 1, -1)
    # t=2: C01(+1), C11(+1)
    set_val(W, 1, 2, 1); set_val(W, 3, 2, 1)
    # t=3: C00(+1), C10(+1)
    set_val(W, 0, 3, 1); set_val(W, 2, 3, 1)
    # t=4: C00(-1), C01(+1)
    set_val(W, 0, 4, -1); set_val(W, 1, 4, 1)
    # t=5: C11(+1)
    set_val(W, 3, 5, 1)
    # t=6: C00(+1)
    set_val(W, 0, 6, 1)

    # Block 2 (B cols 2,3)
    # B indices: j*p+k. j=0,1. k=2,3 -> 2,3,6,7
    # t=7 (M1): b02+b13
    set_val(U, 0, 7, 1); set_val(U, 3, 7, 1)
    set_val(V, 2, 7, 1); set_val(V, 7, 7, 1)
    # t=8 (M2): b02
    set_val(U, 2, 8, 1); set_val(U, 3, 8, 1)
    set_val(V, 2, 8, 1)
    # t=9 (M3): b03-b13
    set_val(U, 0, 9, 1)
    set_val(V, 3, 9, 1); set_val(V, 7, 9, -1)
    # t=10 (M4): b12-b02
    set_val(U, 3, 10, 1)
    set_val(V, 6, 10, 1); set_val(V, 2, 10, -1)
    # t=11 (M5): b13
    set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
    set_val(V, 7, 11, 1)
    # t=12 (M6): b02+b12
    set_val(U, 2, 12, 1); set_val(U, 0, 12, -1)
    set_val(V, 2, 12, 1); set_val(V, 6, 12, 1)
    # t=13 (M7): b03+b13
    set_val(U, 1, 13, 1); set_val(U, 3, 13, -1)
    set_val(V, 3, 13, 1); set_val(V, 7, 13, 1)

    # W for Block 2 (C cols 2,3)
    # C indices: k*n+i. k=2,3. i=0,1 -> 4,5,6,7
    # t=7: C02(+1), C13(+1) -> idx 4, 7
    set_val(W, 4, 7, 1); set_val(W, 7, 7, 1)
    # t=8: C12(+1), C13(-1) -> idx 6, 7
    set_val(W, 6, 8, 1); set_val(W, 7, 8, -1)
    # t=9: C03(+1), C13(+1) -> idx 5, 7
    set_val(W, 5, 9, 1); set_val(W, 7, 9, 1)
    # t=10: C02(+1), C12(+1) -> idx 4, 6
    set_val(W, 4, 10, 1); set_val(W, 6, 10, 1)
    # t=11: C02(-1), C03(+1) -> idx 4, 5
    set_val(W, 4, 11, -1); set_val(W, 5, 11, 1)
    # t=12: C13(+1) -> idx 7
    set_val(W, 7, 12, 1)
    # t=13: C02(+1) -> idx 4
    set_val(W, 4, 13, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
