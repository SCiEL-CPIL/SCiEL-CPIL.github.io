import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitioning
    # N: [2, 1]
    # M: [2, 2, 1]
    # P: [2, 2, 1]
    N_parts = [2, 1]
    M_parts = [2, 2, 1]
    P_parts = [2, 2, 1]
    
    # Calculate total rank
    rank = 0
    for ni in N_parts:
        for mj in M_parts:
            for pk in P_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    rank += 7
                else:
                    rank += ni * mj * pk
                    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen matrices
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Flattened row-major
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, -1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ]
    
    # Iterate over blocks
    I = 0
    for i_idx, ni in enumerate(N_parts):
        J = 0
        for j_idx, mj in enumerate(M_parts):
            K = 0
            for k_idx, pk in enumerate(P_parts):
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen block
                    for c in range(7):
                        # Fill U
                        for ui in range(2):
                            for uj in range(2):
                                u_row = (I + ui) * m + (J + uj)
                                U[u_row][col] = U_S[ui * 2 + uj][c]
                        # Fill V
                        for vj in range(2):
                            for vk in range(2):
                                v_row = (J + vj) * p + (K + vk)
                                V[v_row][col] = V_S[vj * 2 + vk][c]
                        # Fill W
                        for wk in range(2):
                            for wi in range(2):
                                w_row = (K + wk) * n + (I + wi)
                                W[w_row][col] = W_S[wk * 2 + wi][c]
                        col += 1
                else:
                    # Schoolbook block
                    for ui in range(ni):
                        for uj in range(mj):
                            for uk in range(pk):
                                u_row = (I + ui) * m + (J + uj)
                                v_row = (J + uj) * p + (K + uk)
                                w_row = (K + uk) * n + (I + ui)
                                U[u_row][col] = 1
                                V[v_row][col] = 1
                                W[w_row][col] = 1
                                col += 1
                K += pk
            J += mj
        I += ni
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
