import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, 16 for 4x4x2 part
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using double Strassen (rank 49)
    # This is complex to code from scratch correctly in short time.
    # Let's stick to the simpler 84 rank solution which is definitely correct.
    # Or try to code the 65 rank solution carefully.
    
    # Given the risk of bugs in the double Strassen implementation, 
    # and the time limit, the 84 rank solution is safer.
    # However, 65 is a significant improvement.
    
    # Let's try to implement the 65 rank solution.
    
    # Strassen for 2x2
    strassen_alphas = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    strassen_betas = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    strassen_gammas = [
        [[1, 0], [0, 1]], [[0, 0], [1, -1]], [[0, 1], [0, 1]],
        [[1, 0], [1, 0]], [[-1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]
    ]
    
    col = 0
    # Part 1: A (4x4) * B1 (4x4)
    # B1 is B[:, 0:4]
    # Use double Strassen
    # Outer Strassen on 2x2 blocks
    # A blocks A00, A01, A10, A11
    # B1 blocks B00, B01, B10, B11
    # C1 blocks C00, C01, C10, C11
    
    # We need to map the outer Strassen terms to inner Strassen terms.
    # This is effectively a recursive tensor decomposition.
    # Due to complexity, I will stick to the 84 rank solution.
    
    # Re-implementing 84 rank solution
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(2):
        for J in range(2):
            for K in range(3):
                for t in range(7):
                    for p_ in range(2):
                        for q_ in range(2):
                            val = strassen_alphas[t][p_*2+q_]
                            if val != 0:
                                u_idx = (2*I + p_) * 4 + (2*J + q_)
                                U[u_idx][col] = val
                    for r_ in range(2):
                        for s_ in range(2):
                            val = strassen_betas[t][r_*2+s_]
                            if val != 0:
                                v_idx = (2*J + r_) * 6 + (2*K + s_)
                                V[v_idx][col] = val
                    for u_ in range(2):
                        for v_ in range(2):
                            val = strassen_gammas[t][u_][v_]
                            if val != 0:
                                w_idx = (2*K + u_) * 4 + (2*I + v_)
                                W[w_idx][col] = val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
