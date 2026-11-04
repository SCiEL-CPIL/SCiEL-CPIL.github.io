import time

N = 3
M = 4
P = 6

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = N, M, P
    rank = 66

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r_idx = 0

    # Strassen 2x2x2 factors
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_str = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    for k_g in range(3):
        k_off = k_g * 2
        for j_g in range(2):
            j_off = j_g * 2
            
            # Part 1: i in 0,1 (Strassen)
            for i_local in range(2):
                i_global = i_local
                for j_local in range(2):
                    j_global = j_off + j_local
                    for k_local in range(2):
                        k_global = k_off + k_local
                        
                        u_idx = i_local * 2 + j_local
                        v_idx = j_local * 2 + k_local
                        w_idx = k_local * 2 + i_local
                        
                        for r_sub in range(7):
                            U[i_global * m + j_global][r_idx + r_sub] += U_str[u_idx][r_sub]
                            V[j_global * p + k_global][r_idx + r_sub] += V_str[v_idx][r_sub]
                            W[k_global * n + i_global][r_idx + r_sub] += W_str[w_idx][r_sub]
                            
            r_idx += 7

            # Part 2: i in 2 (Schoolbook 1x2x2)
            i_global = 2
            # Terms:
            # 0: A[2][j_off] * B[j_off][k_off] -> C[2][k_off]
            # 1: A[2][j_off] * B[j_off][k_off+1] -> C[2][k_off+1]
            # 2: A[2][j_off+1] * B[j_off+1][k_off] -> C[2][k_off]
            # 3: A[2][j_off+1] * B[j_off+1][k_off+1] -> C[2][k_off+1]
            
            # Term 0
            U[i_global * m + j_off][r_idx + 0] += 1
            V[j_off * p + k_off][r_idx + 0] += 1
            W[k_off * n + i_global][r_idx + 0] += 1
            
            # Term 1
            U[i_global * m + j_off][r_idx + 1] += 1
            V[j_off * p + (k_off + 1)][r_idx + 1] += 1
            W[(k_off + 1) * n + i_global][r_idx + 1] += 1
            
            # Term 2
            U[i_global * m + (j_off + 1)][r_idx + 2] += 1
            V[(j_off + 1) * p + k_off][r_idx + 2] += 1
            W[k_off * n + i_global][r_idx + 2] += 1
            
            # Term 3
            U[i_global * m + (j_off + 1)][r_idx + 3] += 1
            V[(j_off + 1) * p + (k_off + 1)][r_idx + 3] += 1
            W[(k_off + 1) * n + i_global][r_idx + 3] += 1
            
            r_idx += 4

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
