"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # U2[m][u][v] is coeff of X[u][v] in m-th factor
    U2 = [
        [[1, 0], [0, 1]], # m=1: X00+X11
        [[0, 0], [1, 1]], # m=2: X10+X11
        [[1, 0], [0, 0]], # m=3: X00
        [[0, 0], [0, 1]], # m=4: X11
        [[1, 1], [0, 0]], # m=5: X00+X01
        [[-1, 0], [1, 0]],# m=6: X10-X00
        [[0, 1], [0, -1]] # m=7: X01-X11
    ]
    
    V2 = [
        [[1, 0], [0, 1]], # m=1: Y00+Y11
        [[1, 0], [0, 0]], # m=2: Y00
        [[0, 1], [0, -1]],# m=3: Y01-Y11
        [[-1, 0], [1, 0]],# m=4: Y10-Y00
        [[0, 0], [0, 1]], # m=5: Y11
        [[1, 1], [0, 0]], # m=6: Y00+Y01
        [[0, 0], [1, 1]]  # m=7: Y10+Y11
    ]
    
    W2 = [
        [1, 0, 0, 1], # m=1: Z00+=1, Z11+=1 (Z00, Z01, Z10, Z11)
        [0, 0, 1, -1],# m=2: Z10+=1, Z11-=1
        [0, 1, 0, 1], # m=3: Z01+=1, Z11+=1
        [1, 0, 1, 0], # m=4: Z00+=1, Z10+=1
        [-1, 1, 0, 0],# m=5: Z00-=1, Z01+=1
        [0, 0, 0, 1], # m=6: Z11+=1
        [1, 0, 0, 0]  # m=7: Z00+=1
    ]
    
    # Block level coefficients alpha[k][p][q] for X_k = sum alpha A_pq
    # k is 0-indexed 0..6 corresponding to k=1..7
    alpha = [
        [[1, 0], [0, 1]], # k=1: A00+A11
        [[0, 0], [1, 1]], # k=2: A10+A11
        [[1, 0], [0, 0]], # k=3: A00
        [[0, 0], [0, 1]], # k=4: A11
        [[1, 1], [0, 0]], # k=5: A00+A01
        [[-1, 0], [1, 0]],# k=6: A10-A00
        [[0, 1], [0, -1]] # k=7: A01-A11
    ]
    
    beta = [
        [[1, 0], [0, 1]], # k=1: B00+B11
        [[1, 0], [0, 0]], # k=2: B00
        [[0, 1], [0, -1]],# k=3: B01-B11
        [[-1, 0], [1, 0]],# k=4: B10-B00
        [[0, 0], [0, 1]], # k=5: B11
        [[1, 1], [0, 0]], # k=6: B00+B01
        [[0, 0], [1, 1]]  # k=7: B10+B11
    ]
    
    W1 = [
        [1, 0, 0, 1], # k=1: C00+=1, C11+=1
        [0, 0, 1, -1],# k=2: C10+=1, C11-=1
        [0, 1, 0, 1], # k=3: C01+=1, C11+=1
        [1, 0, 1, 0], # k=4: C00+=1, C10+=1
        [-1, 1, 0, 0],# k=5: C00-=1, C01+=1
        [0, 0, 0, 1], # k=6: C11+=1
        [1, 0, 0, 0]  # k=7: C00+=1
    ]
    
    for k in range(7):
        for m in range(7):
            c = k * 7 + m
            # Iterate over all entries in A, B, C
            for i in range(4):
                for j in range(4):
                    # i, j correspond to A[i][j]
                    # i = 2*p + u, j = 2*q + v
                    p = i // 2
                    u = i % 2
                    q = j // 2
                    v = j % 2
                    coeff = alpha[k][p][q] * U2[m][u][v]
                    if coeff != 0:
                        U[i * 4 + j][c] += coeff
                        
            for j in range(4):
                for k_idx in range(4):
                    # j, k_idx correspond to B[j][k_idx]
                    # j = 2*r + u, k_idx = 2*s + v
                    r = j // 2
                    u = j % 2
                    s = k_idx // 2
                    v = k_idx % 2
                    coeff = beta[k][r][s] * V2[m][u][v]
                    if coeff != 0:
                        V[j * 4 + k_idx][c] += coeff
                        
            for k_idx in range(4):
                for i in range(4):
                    # k_idx, i correspond to C[k_idx][i]
                    # k_idx = 2*q + v, i = 2*p + u
                    q = k_idx // 2
                    v = k_idx % 2
                    p = i // 2
                    u = i % 2
                    coeff = W1[k][p * 2 + q] * W2[m][u * 2 + v]
                    if coeff != 0:
                        W[k_idx * 4 + i][c] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
