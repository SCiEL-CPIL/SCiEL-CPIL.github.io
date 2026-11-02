import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Indices
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 1, 0, 1, 0, 1) # C00
    U[1][col-1] += 1 # A11
    V[1*5+1][col-1] += 1 # B11
    W[1*3+1][col-1] += 1 # C11
    
    # M2 = (A10+A11)B00
    add_term(3, 1, 0, 1, 1, 1) # C10
    U[4][col-1] += 1 # A11
    W[1*3+1][col-1] -= 1 # C11 (coeff -1)
    
    # M3 = A00(B01-B11)
    add_term(0, 1, 0*5+1, 1, 1*3+0, 1) # C01
    V[1*5+1][col-1] -= 1 # B11
    W[1*3+1][col-1] += 1 # C11
    
    # M4 = A11(B10-B00)
    add_term(4, 1, 1*5+0, 1, 0, 1) # C00
    V[0][col-1] -= 1 # B00
    W[1][col-1] += 1 # C10
    
    # M5 = (A00+A01)B11
    add_term(0, 1, 1*5+1, 1, 0, -1) # C00
    U[1][col-1] += 1 # A01
    W[1*3+0][col-1] += 1 # C01
    
    # M6 = (A10-A00)(B00+B01)
    add_term(3, 1, 0, 1, 1*3+1, 1) # C11
    U[0][col-1] -= 1 # A00
    V[0*5+1][col-1] += 1 # B01
    
    # M7 = (A01-A11)(B10+B11)
    add_term(1, 1, 1*5+0, 1, 0, 1) # C00
    U[4][col-1] -= 1 # A11
    V[1*5+1][col-1] += 1 # B11
    W[1*3+1][col-1] -= 1 # C11
    
    # Part 2: Schoolbook 2x2x3 for A[0:2,0:2] * B[0:2,2:5]
    # A00 * B0,2..4
    add_term(0, 1, 0*5+2, 1, 2*3+0, 1) # C02
    add_term(0, 1, 0*5+3, 1, 3*3+0, 1) # C03
    add_term(0, 1, 0*5+4, 1, 4*3+0, 1) # C04
    
    # A01 * B1,2..4
    add_term(1, 1, 1*5+2, 1, 2*3+0, 1) # C02
    add_term(1, 1, 1*5+3, 1, 3*3+0, 1) # C03
    add_term(1, 1, 1*5+4, 1, 4*3+0, 1) # C04
    
    # A10 * B0,2..4
    add_term(3, 1, 0*5+2, 1, 2*3+1, 1) # C12
    add_term(3, 1, 0*5+3, 1, 3*3+1, 1) # C13
    add_term(3, 1, 0*5+4, 1, 4*3+1, 1) # C14
    
    # A11 * B1,2..4
    add_term(4, 1, 1*5+2, 1, 2*3+1, 1) # C12
    add_term(4, 1, 1*5+3, 1, 3*3+1, 1) # C13
    add_term(4, 1, 1*5+4, 1, 4*3+1, 1) # C14
    
    # Part 3: v * b_bot (2x1 * 1x5)
    # A02 * B2,0..4
    add_term(2, 1, 2*5+0, 1, 0*3+0, 1) # C00
    add_term(2, 1, 2*5+1, 1, 1*3+0, 1) # C01
    add_term(2, 1, 2*5+2, 1, 2*3+0, 1) # C02
    add_term(2, 1, 2*5+3, 1, 3*3+0, 1) # C03
    add_term(2, 1, 2*5+4, 1, 4*3+0, 1) # C04
    
    # A12 * B2,0..4
    add_term(5, 1, 2*5+0, 1, 0*3+1, 1) # C10
    add_term(5, 1, 2*5+1, 1, 1*3+1, 1) # C11
    add_term(5, 1, 2*5+2, 1, 2*3+1, 1) # C12
    add_term(5, 1, 2*5+3, 1, 3*3+1, 1) # C13
    add_term(5, 1, 2*5+4, 1, 4*3+1, 1) # C14
    
    # Part 4: w^T * B_top (1x2 * 2x5)
    # A20 * B0,0..4
    add_term(6, 1, 0*5+0, 1, 0*3+2, 1) # C20
    add_term(6, 1, 0*5+1, 1, 1*3+2, 1) # C21
    add_term(6, 1, 0*5+2, 1, 2*3+2, 1) # C22
    add_term(6, 1, 0*5+3, 1, 3*3+2, 1) # C23
    add_term(6, 1, 0*5+4, 1, 4*3+2, 1) # C24
    
    # A21 * B1,0..4
    add_term(7, 1, 1*5+0, 1, 0*3+2, 1) # C20
    add_term(7, 1, 1*5+1, 1, 1*3+2, 1) # C21
    add_term(7, 1, 1*5+2, 1, 2*3+2, 1) # C22
    add_term(7, 1, 1*5+3, 1, 3*3+2, 1) # C23
    add_term(7, 1, 1*5+4, 1, 4*3+2, 1) # C24
    
    # Part 5: s * b_bot (1x1 * 1x5)
    # A22 * B2,0..4
    add_term(8, 1, 2*5+0, 1, 0*3+2, 1) # C20
    add_term(8, 1, 2*5+1, 1, 1*3+2, 1) # C21
    add_term(8, 1, 2*5+2, 1, 2*3+2, 1) # C22
    add_term(8, 1, 2*5+3, 1, 3*3+2, 1) # C23
    add_term(8, 1, 2*5+4, 1, 4*3+2, 1) # C24
    
    return {"U": U, "V": V, "W": W}
