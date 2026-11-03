"""Matrix multiplication 3x4 x 4x6 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    terms = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_dict = {(i, j): 1}
                v_dict = {(j, k): 1}
                w_dict = {(i, k): 1}
                terms.append((u_dict, v_dict, w_dict))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (r, mm), val in u_dict.items():
            U[r * m + mm][c] = val
        for (mm, c_idx), val in v_dict.items():
            V[mm * p + c_idx][c] = val
        for (r, c_idx), val in w_dict.items():
            W[c_idx * n + r][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
