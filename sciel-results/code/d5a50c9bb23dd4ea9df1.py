"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 schoolbook and 2x2 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition vectors
    # U2, V2, W2 are lists of 7 vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    V2 = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [-1, 0, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    W2 = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, 1, 0], # M2: C10+1
        [0, 1, 0, 1], # M3: C01+1, C11+1
        [1, 0, 1, 0], # M4: C00+1, C10+1
        [-1, 1, 0, 0],# M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [1, 0, 0, -1] # M7: C00+1, C11-1
    ]
    
    # 3x3 schoolbook loop
    for I in range(3):
        for K in range(3):
            for J in range(3):
                t = I * 9 + K * 3 + J
                for s in range(7):
                    c = t * 7 + s
                    # Fill U
                    for il in range(2):
                        for jl in range(2):
                            i = I * 2 + il
                            j = K * 2 + jl
                            U[i * 6 + j][c] = U2[s][il * 2 + jl]
                            
                    # Fill V
                    for jl in range(2):
                        for kl in range(2):
                            j = K * 2 + jl
                            k = J * 2 + kl
                            V[j * 6 + k][c] = V2[s][jl * 2 + kl]
                            
                    # Fill W
                    for kl in range(2):
                        for il in range(2):
                            k = J * 2 + kl
                            i = I * 2 + il
                            W[k * 6 + i][c] = W2[s][kl * 2 + il]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
