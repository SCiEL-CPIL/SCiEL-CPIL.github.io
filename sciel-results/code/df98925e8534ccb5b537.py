import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank is 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of p=2
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Part 1: A_1 (2x1) * B_1 (1x2) -> Rank 4
        # A_1 uses j=0. B_1 uses j=0.
        # U rows: (0,0), (1,0)
        # V rows: (0,k0), (0,k1)
        # W rows: (k0,0), (k0,1), (k1,0), (k1,1)
        
        # Term 1: a[0,0] * b[0,k0] -> C[0,k0]
        U[0*3 + 0][col] = 1
        V[0*10 + k0][col] = 1
        W[k0*2 + 0][col] = 1
        col += 1
        
        # Term 2: a[0,0] * b[0,k1] -> C[0,k1]
        U[0*3 + 0][col] = 1
        V[0*10 + k1][col] = 1
        W[k1*2 + 0][col] = 1
        col += 1
        
        # Term 3: a[1,0] * b[0,k0] -> C[1,k0]
        U[1*3 + 0][col] = 1
        V[0*10 + k0][col] = 1
        W[k0*2 + 1][col] = 1
        col += 1
        
        # Term 4: a[1,0] * b[0,k1] -> C[1,k1]
        U[1*3 + 0][col] = 1
        V[0*10 + k1][col] = 1
        W[k1*2 + 1][col] = 1
        col += 1
        
        # Part 2: A_2 (2x2) * B_2 (2x2) -> Rank 7 (Strassen)
        # A_2 uses j=1,2. B_2 uses j=1,2.
        # A_2 = [[a01, a02], [a11, a12]]
        # B_2 = [[b1k0, b1k1], [b2k0, b2k1]]
        
        # Strassen factors:
        # P1 = (a+d)(e+f)
        # P2 = (c+d)e
        # P3 = (a-b)f
        # P4 = d(f-g)
        # P5 = a(e-h)
        # P6 = (a+b)h
        # P7 = (c-d)g
        
        # U indices for A_2:
        # a = U[0*3+1], b = U[0*3+2], c = U[1*3+1], d = U[1*3+2]
        # V indices for B_2:
        # e = V[1*10+k0], f = V[1*10+k1], g = V[2*10+k0], h = V[2*10+k1]
        
        # P1
        U[0*3+1][col] = 1
        U[1*3+2][col] = 1
        V[1*10+k0][col] = 1
        V[1*10+k1][col] = 1
        # C contributions: C[0,k0] += P1, C[0,k1] -= P4 (wait, P1 is in C00 and C11)
        # C00 = P1 - P4 + P5 - P7
        # C01 = P2 + P4
        # C10 = P3 + P5
        # C11 = P1 - P2 + P3 + P6
        
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # P2
        U[1*3+1][col] = 1
        U[1*3+2][col] = 1
        V[1*10+k0][col] = 1
        W[k0*2+0][col] = -1 # C01 = P2 + P4, so P2 goes to C01? No.
        # Correction:
        # C01 = P2 + P4. So P2 contributes to C[0, k1]?
        # C is 2x2. C[0,0], C[0,1], C[1,0], C[1,1].
        # C[0,1] corresponds to W[k0*2+1] and W[k1*2+1]?
        # W index is k*n + i.
        # C[0, k0] is row 0, col k0. W index k0*2 + 0.
        # C[0, k1] is row 0, col k1. W index k1*2 + 0.
        # C[1, k0] is row 1, col k0. W index k0*2 + 1.
        # C[1, k1] is row 1, col k1. W index k1*2 + 1.
        
        # Strassen Output:
        # C00 = P1 - P4 + P5 - P7
        # C01 = P2 + P4
        # C10 = P3 + P5
        # C11 = P1 - P2 + P3 + P6
        
        # So:
        # P1 -> C00 (+1), C11 (+1)
        # P2 -> C01 (+1), C11 (-1)
        # P3 -> C10 (+1), C11 (+1)
        # P4 -> C00 (-1), C01 (+1)
        # P5 -> C00 (+1), C10 (+1)
        # P6 -> C11 (+1)
        # P7 -> C00 (-1)
        
        # Let's re-assign W for P1
        # W[k0*2+0][col-1] = 1 (C00)
        # W[k1*2+1][col-1] = 1 (C11)
        # This was done above.
        
        # P2
        U[1*3+1][col] = 1 # c
        U[1*3+2][col] = 1 # d
        V[1*10+k0][col] = 1 # e
        W[k1*2+0][col] = 1 # C01
        W[k1*2+1][col] = -1 # C11
        col += 1
        
        # P3
        U[0*3+1][col] = 1 # a
        U[0*3+2][col] = -1 # b
        V[1*10+k1][col] = 1 # f
        W[k0*2+1][col] = 1 # C10
        W[k1*2+1][col] = 1 # C11
        col += 1
        
        # P4
        U[1*3+2][col] = 1 # d
        V[1*10+k1][col] = 1 # f
        V[2*10+k0][col] = -1 # g
        W[k0*2+0][col] = -1 # C00
        W[k1*2+0][col] = 1 # C01
        col += 1
        
        # P5
        U[0*3+1][col] = 1 # a
        V[1*10+k0][col] = 1 # e
        V[2*10+k1][col] = -1 # h
        W[k0*2+0][col] = 1 # C00
        W[k0*2+1][col] = 1 # C10
        col += 1
        
        # P6
        U[0*3+1][col] = 1 # a
        U[0*3+2][col] = 1 # b
        V[2*10+k1][col] = 1 # h
        W[k1*2+1][col] = 1 # C11
        col += 1
        
        # P7
        U[1*3+1][col] = 1 # c
        U[1*3+2][col] = -1 # d
        V[2*10+k0][col] = 1 # g
        W[k0*2+0][col] = -1 # C00
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
