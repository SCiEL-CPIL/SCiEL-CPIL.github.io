import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Strassen decomposition for 2x2x2
    # U_A: 4x7, V_A: 4x7, W_A: 4x7
    U_A = [
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    V_A = [
        [1, 1, 0, -1, 0, 0, 1], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 1, 1], # g
        [1, 0, -1, 0, 1, 0, 0] # h
    ]
    W_A = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0] # C22
    ]
    
    # Schoolbook decomposition for 1x2x2
    # U_B: 2x4, V_B: 4x4, W_B: 2x4
    U_B = [
        [1, 0, 1, 0], # u1
        [0, 1, 0, 1]  # u2
    ]
    V_B = [
        [1, 0, 0, 0], # v11 (j=0, k=0)
        [0, 1, 0, 0], # v12 (j=0, k=1)
        [0, 0, 1, 0], # v21 (j=1, k=0)
        [0, 0, 0, 1]  # v22 (j=1, k=1)
    ]
    W_B = [
        [1, 1, 0, 0], # w1
        [0, 0, 1, 1]  # w2
    ]
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(4):
            c = s * 4 + t
            
            # Fill U
            # U index: i * 4 + j
            # i in 0..1, j in 0..3
            # j_A = j // 2, j_B = j % 2
            # U_A index: i * 2 + j_A
            for i in range(2):
                for j in range(4):
                    j_A = j // 2
                    j_B = j % 2
                    u_val = U_A[i * 2 + j_A][s] * U_B[j_B][t]
                    U[i * 4 + j][c] = u_val
                    
            # Fill V
            # V index: j * 4 + k
            # j in 0..3, k in 0..3
            # j_A = j // 2, j_B = j % 2
            # k_A = k // 2, k_B = k % 2
            # V_A index: j_A * 2 + k_A
            # V_B index: j_B * 2 + k_B
            for j in range(4):
                for k in range(4):
                    j_A = j // 2
                    j_B = j % 2
                    k_A = k // 2
                    k_B = k % 2
                    v_val = V_A[j_A * 2 + k_A][s] * V_B[j_B * 2 + k_B][t]
                    V[j * 4 + k][c] = v_val
                    
            # Fill W
            # W index: k * 2 + i
            # k in 0..3, i in 0..1
            # k_A = k // 2, k_B = k % 2
            # W_A index: k_A * 2 + i
            # W_B index: k_B
            for k in range(4):
                for i in range(2):
                    k_A = k // 2
                    k_B = k % 2
                    w_val = W_A[k_A * 2 + i][s] * W_B[k_B][t]
                    W[k * 2 + i][c] = w_val
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
