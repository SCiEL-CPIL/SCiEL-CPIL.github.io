import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 77
    # Outer rank: 11, Inner rank: 7
    outer_rank = 11
    inner_rank = 7
    total_rank = outer_rank * inner_rank
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Inner Strassen decomposition for 2x2 * 2x2
    # U_in, V_in, W_in are 2x7
    U_in = np.zeros((2, inner_rank), dtype=float)
    V_in = np.zeros((2, inner_rank), dtype=float)
    W_in = np.zeros((2, inner_rank), dtype=float)
    
    # Strassen terms:
    # 1: (a+d)(e+i) -> U: (1,1), V: (1,1), W: (1,1)
    U_in[0,0] = 1; U_in[1,0] = 1
    V_in[0,0] = 1; V_in[1,0] = 1
    W_in[0,0] = 1; W_in[1,0] = 1
    
    # 2: (c+d)(e+h) -> U: (0,1), V: (1,1), W: (0,1)
    U_in[0,1] = 0; U_in[1,1] = 1
    V_in[0,1] = 1; V_in[1,1] = 1
    W_in[0,1] = 0; W_in[1,1] = 1
    
    # 3: a(f-i) -> U: (1,0), V: (1,-1), W: (1,0)
    U_in[0,2] = 1; U_in[1,2] = 0
    V_in[0,2] = 1; V_in[1,2] = -1
    W_in[0,2] = 1; W_in[1,2] = 0
    
    # 4: d(h-e) -> U: (0,1), V: (-1,1), W: (0,1)
    U_in[0,3] = 0; U_in[1,3] = 1
    V_in[0,3] = -1; V_in[1,3] = 1
    W_in[0,3] = 0; W_in[1,3] = 1
    
    # 5: (a+b)i -> U: (1,1), V: (0,1), W: (1,0)
    U_in[0,4] = 1; U_in[1,4] = 1
    V_in[0,4] = 0; V_in[1,4] = 1
    W_in[0,4] = 1; W_in[1,4] = 0
    
    # 6: (c-a)(e+f) -> U: (0,-1), V: (1,1), W: (0,1)
    U_in[0,5] = 0; U_in[1,5] = -1
    V_in[0,5] = 1; V_in[1,5] = 1
    W_in[0,5] = 0; W_in[1,5] = 1
    
    # 7: (b-d)(h+i) -> U: (0,-1), V: (1,1), W: (1,0)
    U_in[0,6] = 0; U_in[1,6] = -1
    V_in[0,6] = 1; V_in[1,6] = 1
    W_in[0,6] = 1; W_in[1,6] = 0
    
    # Outer decomposition for 2x2 * 2x3 (block level)
    # U_out (2x11), V_out (2x11), W_out (3x11)
    U_out = np.zeros((2, outer_rank), dtype=float)
    V_out = np.zeros((2, outer_rank), dtype=float)
    W_out = np.zeros((3, outer_rank), dtype=float)
    
    # Terms 0-6: Strassen for K=0,1
    # Copy inner Strassen structure to outer
    for t in range(7):
        U_out[:, t] = U_in[:, t]
        V_out[:, t] = V_in[:, t]
        W_out[0, t] = W_in[0, t]
        W_out[1, t] = W_in[1, t]
        W_out[2, t] = 0
        
    # Terms 7-10: Schoolbook for K=2
    # C_02 = A_00 B_02 + A_01 B_12
    # C_12 = A_10 B_02 + A_11 B_12
    
    # Term 7: A_00 B_02
    U_out[0, 7] = 1; U_out[1, 7] = 0
    V_out[0, 7] = 1; V_out[1, 7] = 0
    W_out[0, 7] = 0; W_out[1, 7] = 0; W_out[2, 7] = 1
    
    # Term 8: A_01 B_12
    U_out[0, 8] = 1; U_out[1, 8] = 0
    V_out[0, 8] = 0; V_out[1, 8] = 1
    W_out[0, 8] = 0; W_out[1, 8] = 0; W_out[2, 8] = 1
    
    # Term 9: A_10 B_02
    U_out[0, 9] = 0; U_out[1, 9] = 1
    V_out[0, 9] = 1; V_out[1, 9] = 0
    W_out[0, 9] = 0; W_out[1, 9] = 0; W_out[2, 9] = 1
    
    # Term 10: A_11 B_12
    U_out[0, 10] = 0; U_out[1, 10] = 1
    V_out[0, 10] = 0; V_out[1, 10] = 1
    W_out[0, 10] = 0; W_out[1, 10] = 0; W_out[2, 10] = 1
    
    # Combine
    for t in range(outer_rank):
        for s in range(inner_rank):
            c = t * inner_rank + s
            for I in range(2):
                for i_prime in range(2):
                    i = I * 2 + i_prime
                    U[i, c] = U_out[I, t] * U_in[i_prime, s]
            for J in range(2):
                for j_prime in range(2):
                    j = J * 2 + j_prime
                    V[j, c] = V_out[J, t] * V_in[j_prime, s]
            for K in range(3):
                for k_prime in range(2):
                    k = K * 2 + k_prime
                    W[k, c] = W_out[K, t] * W_in[k_prime, s]
                    
    # Convert to list of lists as required
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
