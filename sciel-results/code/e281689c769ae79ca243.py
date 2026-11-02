"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank is 7 * (p/2) = 7 * 8 = 56
    rank = 7 * (p // 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen factors for 2x2 multiplication
    # Indices: s in 0..6
    # U_s[i,j], V_s[j,h], W_s[h,i]
    # h is local column index (0 or 1)
    
    # s=0: M1 = (a00+a11)(b00+b11)
    # c00 += 1, c11 -= 1
    U0 = [[1, 0], [0, 1]]
    V0 = [[1, 0], [0, 1]]
    W0 = [[1, 0], [0, -1]] # W[h,i] -> W[0,0]=1, W[1,1]=-1
    
    # s=1: M2 = (a10+a11)b00
    # c10 += 1, c11 -= 1
    U1 = [[0, 0], [1, 1]]
    V1 = [[1, 0], [0, 0]]
    W1 = [[0, 1], [0, -1]] # W[0,1]=1, W[1,1]=-1
    
    # s=2: M3 = a00(b10-b11)
    # c01 += 1, c11 += 1
    U2 = [[1, 0], [0, 0]]
    V2 = [[0, 0], [1, -1]]
    W2 = [[0, 0], [1, 1]] # W[1,0]=1, W[1,1]=1
    
    # s=3: M4 = a11(b01+b11)
    # c00 += 1
    U3 = [[0, 0], [0, 1]]
    V3 = [[0, 1], [0, 1]]
    W3 = [[1, 0], [0, 0]] # W[0,0]=1
    
    # s=4: M5 = (a00+a10)b11
    # c00 -= 1, c11 += 1
    U4 = [[1, 0], [1, 0]]
    V4 = [[0, 0], [0, 1]]
    W4 = [[-1, 0], [0, 1]] # W[0,0]=-1, W[1,1]=1
    
    # s=5: M6 = (a11-a00)(b00+b10)
    # c10 += 1
    U5 = [[-1, 0], [0, 1]]
    V5 = [[1, 0], [1, 0]]
    W5 = [[0, 1], [0, 0]] # W[0,1]=1
    
    # s=6: M7 = (a01-a11)(b01+b11)
    # c01 -= 1, c11 += 1
    U6 = [[0, 1], [0, -1]]
    V6 = [[0, 1], [0, 1]]
    W6 = [[0, -1], [0, 1]] # W[1,0]=-1, W[1,1]=1
    
    Us = [U0, U1, U2, U3, U4, U5, U6]
    Vs = [V0, V1, V2, V3, V4, V5, V6]
    Ws = [W0, W1, W2, W3, W4, W5, W6]
    
    num_blocks = p // 2
    
    for g in range(num_blocks):
        for s in range(7):
            r = 7 * g + s
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = Us[s][i][j]
                    
            # Fill V
            for j in range(m):
                for h in range(2):
                    k = 2 * g + h
                    V[j * p + k][r] = Vs[s][j][h]
                    
            # Fill W
            for h in range(2):
                k = 2 * g + h
                for i in range(n):
                    W[k * n + i][r] = Ws[s][h][i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
