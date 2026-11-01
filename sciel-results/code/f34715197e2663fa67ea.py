import time

def construct():
    n, m, p = 2, 2, 14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for each of the 7 products M1..M7
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block t: 2t:(0,2t), 2t+1:(0,2t+1), 14+2t:(1,2t), 15+2t:(1,2t+1)
    # W indices for block t: 4t:(0,2t), 4t+1:(1,2t), 4t+2:(0,2t+1), 4t+3:(1,2t+1)
    
    # M1: (a00+a11)(b00+b11) -> c00+, c11+
    # M2: (a10+a11)b00 -> c10+, c11-
    # M3: a00(b01-b11) -> c01+, c11+
    # M4: a11(b10-b00) -> c00+, c10+
    # M5: (a00+a01)b11 -> c00-, c01+
    # M6: (a10-a00)(b00+b01) -> c11+
    # M7: (a01-a11)(b10+b11) -> c00+
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V contributions are sparse per block
    # M1: b00, b11 -> idx 2t, 14+2t
    # M2: b00 -> idx 2t
    # M3: b01, -b11 -> idx 2t+1, -(15+2t)
    # M4: b10, -b00 -> idx 14+2t, -(2t)
    # M5: b11 -> idx 15+2t
    # M6: b00, b01 -> idx 2t, 2t+1
    # M7: b10, b11 -> idx 14+2t, 15+2t
    
    # W contributions
    # M1: c00, c11 -> idx 4t, 4t+3
    # M2: c10, -c11 -> idx 4t+1, -(4t+3)
    # M3: c01, c11 -> idx 4t+2, 4t+3
    # M4: c00, c10 -> idx 4t, 4t+1
    # M5: -c00, c01 -> idx -(4t), 4t+2
    # M6: c11 -> idx 4t+3
    # M7: c00 -> idx 4t

    for t in range(7):
        base_v0 = 2 * t
        base_v1 = 14 + 2 * t
        base_w = 4 * t
        
        for q in range(7):
            col = t * 7 + q
            
            # Fill U
            for i in range(4):
                U[i][col] = strassen_u[q][i]
            
            # Fill V
            if q == 0:
                V[base_v0][col] = 1
                V[base_v1 + 1][col] = 1 # 14+2t+1
            elif q == 1:
                V[base_v0][col] = 1
            elif q == 2:
                V[base_v0 + 1][col] = 1
                V[base_v1 + 1][col] = -1
            elif q == 3:
                V[base_v1][col] = 1
                V[base_v0][col] = -1
            elif q == 4:
                V[base_v1 + 1][col] = 1
            elif q == 5:
                V[base_v0][col] = 1
                V[base_v0 + 1][col] = 1
            elif q == 6:
                V[base_v1][col] = 1
                V[base_v1 + 1][col] = 1
            
            # Fill W
            if q == 0:
                W[base_w][col] = 1
                W[base_w + 3][col] = 1
            elif q == 1:
                W[base_w + 1][col] = 1
                W[base_w + 3][col] = -1
            elif q == 2:
                W[base_w + 2][col] = 1
                W[base_w + 3][col] = 1
            elif q == 3:
                W[base_w][col] = 1
                W[base_w + 1][col] = 1
            elif q == 4:
                W[base_w][col] = -1
                W[base_w + 2][col] = 1
            elif q == 5:
                W[base_w + 3][col] = 1
            elif q == 6:
                W[base_w][col] = 1

    return {"U": U, "V": V, "W": W}
