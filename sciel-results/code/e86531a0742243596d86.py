import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col, a, b, c, d, e, f, g, h, w00, w01, w10, w11):
        # M1 = (a+d)(e+h)
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        
        # M2 = (c+d)e
        U[c][col+1] += 1
        U[d][col+1] += 1
        V[e][col+1] += 1
        W[w10][col+1] += 1
        W[w11][col+1] -= 1
        
        # M3 = a(f-h)
        U[a][col+2] += 1
        V[f][col+2] += 1
        V[h][col+2] -= 1
        W[w01][col+2] += 1
        W[w11][col+2] += 1
        
        # M4 = d(g-e)
        U[d][col+3] += 1
        V[g][col+3] += 1
        V[e][col+3] -= 1
        W[w00][col+3] += 1
        W[w10][col+3] += 1
        
        # M5 = (a+b)h
        U[a][col+4] += 1
        U[b][col+4] += 1
        V[h][col+4] += 1
        W[w00][col+4] -= 1
        W[w01][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[c][col+5] += 1
        U[a][col+5] -= 1
        V[e][col+5] += 1
        V[f][col+5] += 1
        W[w11][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[b][col+6] += 1
        U[d][col+6] -= 1
        V[g][col+6] += 1
        V[h][col+6] += 1
        W[w00][col+6] += 1
        W[w11][col+6] -= 1

    # Block 1: A_L B_11
    # A_L: 0,1,4,5
    # B_11: 0,1,4,5
    # C_11: 0,2,1,3
    add_strassen_block(0, 0, 1, 4, 5, 0, 1, 4, 5, 0, 2, 1, 3)
    
    # Block 2: A_R B_21
    # A_R: 2,3,6,7
    # B_21: 8,9,12,13
    # C_21: 0,2,1,3
    add_strassen_block(7, 2, 3, 6, 7, 8, 9, 12, 13, 0, 2, 1, 3)
    
    # Block 3: A_L B_12
    # A_L: 0,1,4,5
    # B_12: 2,3,6,7
    # C_12: 4,6,5,7
    add_strassen_block(14, 0, 1, 4, 5, 2, 3, 6, 7, 4, 6, 5, 7)
    
    # Block 4: A_R B_22
    # A_R: 2,3,6,7
    # B_22: 10,11,14,15
    # C_22: 4,6,5,7
    add_strassen_block(21, 2, 3, 6, 7, 10, 11, 14, 15, 4, 6, 5, 7)
    
    return {"U": U, "V": V, "W": W}
