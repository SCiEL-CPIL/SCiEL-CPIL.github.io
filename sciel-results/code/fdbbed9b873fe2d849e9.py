import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8

    # Rank-7 Strassen forms for 2x2 multiplication
    # U coefficients (for A): rows = (i, j) of 2x2 A, cols = 7 products
    x_forms = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # a1
        [0, 0, 0, 0, 1,  0, 1],   # a2
        [0, 1, 0, 0, 0,  1, 0],   # a3
        [1, 1, 0, 1, 0,  0, -1]   # a4
    ])
    
    # V coefficients (for B): rows = (j, k) of 2x2 B, cols = 7 products
    y_forms = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # b1
        [0, 0, 1,  0, 0, 1, 0],   # b2
        [0, 0, 0,  1, 0, 0, 1],   # b3
        [1, 0, -1, 0, 1, 0, 1]    # b4
    ])
    
    # W coefficients (for C output): rows = (k, i) of 2x2 C transposed convention, cols = 7 products
    # C[0,0] = m1 - m4 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    # Convention: W[k*n + i, r]. For 2x2, k in {0,1}, i in {0,1}.
    # W index 0: k=0, i=0 -> C[0,0]
    # W index 1: k=0, i=1 -> C[0,1]
    # W index 2: k=1, i=0 -> C[1,0]
    # W index 3: k=1, i=1 -> C[1,1]
    z_forms = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1]
    ])

    # Decomposition structure:
    # A is 2x4. Split columns into two 2-column blocks: A_L (cols 0,1), A_R (cols 2,3).
    # B is 4x8. Split rows into two 2-row blocks: B_Top (rows 0,1), B_Bot (rows 2,3).
    #                 Split columns into four 2-column blocks: B_C0, B_C1, B_C2, B_C3.
    # 
    # C = A * B = A_L * B_Left + A_R * B_Right
    # B_Left uses B rows 0,1 (B_Top) and B rows 2,3 (B_Bot) ? No.
    # B is 4x8. Rows 0,1 multiply with A_L cols 0,1. Rows 2,3 multiply with A_R cols 2,3.
    # C = A_L (2x2) * B_{0:2, 0:8} (2x8) + A_R (2x2) * B_{2:4, 0:8} (2x8)
    #
    # B_{0:2, 0:8} is a 2x8 matrix. We can split it into four 2x2 blocks: B_{0:2, 0:2}, B_{0:2, 2:4}, B_{0:2, 4:6}, B_{0:2, 6:8}.
    # B_{2:4, 0:8} is a 2x8 matrix. We can split it into four 2x2 blocks: B_{2:4, 0:2}, B_{2:4, 2:4}, B_{2:4, 4:6}, B_{2:4, 6:8}.
    #
    # So we have 2 (A blocks) * 4 (B col blocks) = 8 independent 2x2 * 2x2 multiplications.
    # Each uses rank 7. Total rank = 8 * 7 = 56.

    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    r = 0
    # Iterate over A blocks (b_a in {0,1}) and B column groups (g in {0,1,2,3})
    # b_a=0 -> A cols 0,1, B rows 0,1
    # b_a=1 -> A cols 2,3, B rows 2,3
    # g     -> B cols 2*g, 2*g+1
    
    for b_a in range(2):
        for g in range(4):
            # A block: rows 0,1; cols 2*b_a, 2*b_a+1
            # B block: rows 2*b_a, 2*b_a+1; cols 2*g, 2*g+1
            
            for s in range(7):
                # Fill U
                # U index = i * m + j. i in {0,1}, j in {2*b_a, 2*b_a+1}
                for i in range(2):
                    for j_local in range(2):
                        j_global = 2 * b_a + j_local
                        u_idx = i * m + j_global
                        U[u_idx, r] = x_forms[i, j_local, s]
                
                # Fill V
                # V index = j * p + k. j in {2*b_a, 2*b_a+1}, k in {2*g, 2*g+1}
                for j_local in range(2):
                    j_global = 2 * b_a + j_local
                    for k_local in range(2):
                        k_global = 2 * g + k_local
                        v_idx = j_global * p + k_global
                        V[v_idx, r] = y_forms[j_local, k_local, s]
                
                # Fill W
                # W index = k * n + i. k in {2*g, 2*g+1}, i in {0,1}
                for k_local in range(2):
                    k_global = 2 * g + k_local
                    for i in range(2):
                        w_idx = k_global * n + i
                        W[w_idx, r] = z_forms[k_local, i, s]
                
                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
