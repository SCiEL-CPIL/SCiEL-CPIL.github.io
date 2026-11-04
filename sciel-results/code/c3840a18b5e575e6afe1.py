import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    # Total Rank 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base Strassen 2x2 terms (U, V, W vectors of length 4)
    # U index: i*2+j (00, 01, 10, 11)
    # V index: j*2+k (00, 01, 10, 11)
    # W index: k*2+i (00, 10, 01, 11) -> Wait, W index is k*n+i. For n=2, k*2+i.
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    # So W vector order: [C00, C10, C01, C11]
    
    base_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    base_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    base_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, -1], # M2: C10, -C11
        [0, 0, 1, 1], # M3: C01, C11
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0], # M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Block Strassen Coefficients for 4x4
    # s=0..6
    # alpha[s][i_b][j_b]
    # beta[s][j_b][k_b]
    # delta[s][i_b][k_b]
    
    alpha = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    beta = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    delta = [
        [[1, 0], [0, 1]], # M1: C00, C11
        [[0, 0], [1, -1]], # M2: C10, -C11
        [[0, 1], [0, 1]], # M3: C01, C11
        [[1, 0], [1, 0]], # M4: C00, C10
        [[-1, 1], [0, 0]], # M5: -C00, C01
        [[0, 0], [0, 1]], # M6: C11
        [[1, 0], [0, 0]]  # M7: C00
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Generate U, V, W for this term
            for i in range(4):
                for j in range(4):
                    i_b = i // 2
                    i_l = i % 2
                    j_b = j // 2
                    j_l = j % 2
                    U[i * 4 + j][col] = alpha[s][i_b][j_b] * base_U[t][i_l * 2 + j_l]
            
            for j in range(4):
                for k in range(4):
                    j_b = j // 2
                    j_l = j % 2
                    k_b = k // 2
                    k_l = k % 2
                    V[j * 4 + k][col] = beta[s][j_b][k_b] * base_V[t][j_l * 2 + k_l]
            
            for k in range(4):
                for i in range(4):
                    k_b = k // 2
                    k_l = k % 2
                    i_b = i // 2
                    i_l = i % 2
                    W[k * 4 + i][col] = delta[s][i_b][k_b] * base_W[t][k_l * 2 + i_l]
            
            col += 1
    
    # Part 2: 4x4x1 Schoolbook
    # k is always 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 1 + 0][col] = 1
            W[0 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
