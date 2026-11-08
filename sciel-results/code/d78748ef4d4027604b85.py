"""Matrix multiplication 2,4,8 - Strassen-based rank-56 decomposition."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # 2x2 Strassen terms: (U_local, V_local, W_local)
    # U_local indexed by (i_l*2+j_l), V_local by (j_l*2+k_l), W_local by (k_l*2+i_l)
    # W index: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1: z00+=1, z11+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2: z10+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3: z01+=1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4: z00+=1, z10+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5: z00-=1, z01+=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6: z11+=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7: z00+=1
    ]

    col = 0
    for ba in range(2):       # which A/B row block (0 or 1)
        for bb in range(4):   # which column block (0,1,2,3)
            for t in range(7):
                U_loc, V_loc, W_loc = strassen_terms[t]
                for i_l in range(2):
                    for j_l in range(2):
                        u_idx = i_l * m + (2 * ba + j_l)
                        U[u_idx][col] = U_loc[i_l * 2 + j_l]
                for j_l in range(2):
                    for k_l in range(2):
                        v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                        V[v_idx][col] = V_loc[j_l * 2 + k_l]
                for k_l in range(2):
                    for i_l in range(2):
                        w_idx = (2 * bb + k_l) * n + i_l
                        W[w_idx][col] = W_loc[k_l * 2 + i_l]
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
