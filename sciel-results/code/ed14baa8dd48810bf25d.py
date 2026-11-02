import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, idx, col, val):
        mat[idx][col] += val

    # --- Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] (Terms 0-6) ---
    # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 5:(1,0), 6:(1,1)
    # C indices: 0:(0,0), 2:(0,1), 1:(1,0), 3:(1,1)
    
    # M0: (a+d)(e+h)
    set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 6, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
    
    # M1: (c+d)e
    set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)
    
    # M2: a(f-h)
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 6, 2, -1)
    set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)
    
    # M3: d(g-e)
    set_val(U, 4, 3, 1)
    set_val(V, 5, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)
    
    # M4: (a+b)h
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 6, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)
    
    # M5: (c-a)(e+f)
    set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)
    
    # M6: (b-d)(g+h)
    set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
    set_val(V, 5, 6, 1); set_val(V, 6, 6, 1)
    set_val(W, 0, 6, 1)
    
    # --- Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] (Terms 7-13) ---
    # B indices: 2:(0,2), 3:(0,3), 7:(1,2), 8:(1,3)
    # C indices: 4:(0,2), 6:(0,3), 5:(1,2), 7:(1,3)
    
    # M7: (a+d)(u+x)
    set_val(U, 0, 7, 1); set_val(U, 4, 7, 1)
    set_val(V, 2, 7, 1); set_val(V, 7, 7, 1)
    set_val(W, 4, 7, 1); set_val(W, 7, 7, 1)
    
    # M8: (c+d)u
    set_val(U, 3, 8, 1); set_val(U, 4, 8, 1)
    set_val(V, 2, 8, 1)
    set_val(W, 5, 8, 1); set_val(W, 7, 8, -1)
    
    # M9: a(v-x)
    set_val(U, 0, 9, 1)
    set_val(V, 3, 9, 1); set_val(V, 7, 9, -1)
    set_val(W, 6, 9, 1); set_val(W, 7, 9, 1)
    
    # M10: d(x-u)
    set_val(U, 4, 10, 1)
    set_val(V, 7, 10, 1); set_val(V, 2, 10, -1)
    set_val(W, 4, 10, 1); set_val(W, 5, 10, 1)
    
    # M11: (a+b)v
    set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
    set_val(V, 3, 11, 1)
    set_val(W, 4, 11, -1); set_val(W, 6, 11, 1)
    
    # M12: (c-a)(u+v)
    set_val(U, 3, 12, 1); set_val(U, 0, 12, -1)
    set_val(V, 2, 12, 1); set_val(V, 3, 12, 1)
    set_val(W, 7, 12, 1)
    
    # M13: (b-d)(x+y)
    set_val(U, 1, 13, 1); set_val(U, 4, 13, -1)
    set_val(V, 7, 13, 1); set_val(V, 8, 13, 1)
    set_val(W, 4, 13, 1)
    
    # --- Part 3: Schoolbook for A[0:2, 0:2] * B[0:2, 4] (Terms 14-17) ---
    # B indices: 4:(0,4), 9:(1,4)
    # C indices: 8:(0,4), 9:(1,4)
    
    # M14: a*w
    set_val(U, 0, 14, 1)
    set_val(V, 4, 14, 1)
    set_val(W, 8, 14, 1)
    
    # M15: b*w
    set_val(U, 1, 15, 1)
    set_val(V, 4, 15, 1)
    set_val(W, 9, 15, 1)
    
    # M16: c*z
    set_val(U, 3, 16, 1)
    set_val(V, 9, 16, 1)
    set_val(W, 8, 16, 1)
    
    # M17: d*z
    set_val(U, 4, 17, 1)
    set_val(V, 9, 17, 1)
    set_val(W, 9, 17, 1)
    
    # --- Part 4: Schoolbook for A[0:2, 2] * B[2, 0:5] (Terms 18-27) ---
    # A indices: 2:(0,2), 5:(1,2)
    # B indices: 10:(2,0), 11:(2,1), 12:(2,2), 13:(2,3), 14:(2,4)
    
    # M18: A02 * B20 -> C00
    set_val(U, 2, 18, 1)
    set_val(V, 10, 18, 1)
    set_val(W, 0, 18, 1)
    
    # M19: A02 * B21 -> C01
    set_val(U, 2, 19, 1)
    set_val(V, 11, 19, 1)
    set_val(W, 2, 19, 1)
    
    # M20: A02 * B22 -> C02
    set_val(U, 2, 20, 1)
    set_val(V, 12, 20, 1)
    set_val(W, 4, 20, 1)
    
    # M21: A02 * B23 -> C03
    set_val(U, 2, 21, 1)
    set_val(V, 13, 21, 1)
    set_val(W, 6, 21, 1)
    
    # M22: A02 * B24 -> C04
    set_val(U, 2, 22, 1)
    set_val(V, 14, 22, 1)
    set_val(W, 8, 22, 1)
    
    # M23: A12 * B20 -> C10
    set_val(U, 5, 23, 1)
    set_val(V, 10, 23, 1)
    set_val(W, 1, 23, 1)
    
    # M24: A12 * B21 -> C11
    set_val(U, 5, 24, 1)
    set_val(V, 11, 24, 1)
    set_val(W, 3, 24, 1)
    
    # M25: A12 * B22 -> C12
    set_val(U, 5, 25, 1)
    set_val(V, 12, 25, 1)
    set_val(W, 5, 25, 1)
    
    # M26: A12 * B23 -> C13
    set_val(U, 5, 26, 1)
    set_val(V, 13, 26, 1)
    set_val(W, 7, 26, 1)
    
    # M27: A12 * B24 -> C14
    set_val(U, 5, 27, 1)
    set_val(V, 14, 27, 1)
    set_val(W, 9, 27, 1)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
