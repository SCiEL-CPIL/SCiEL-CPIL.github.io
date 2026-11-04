"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Partition j (size m=5) into groups: [0:2], [2:4], [4:5]
    # Partition k (size p=6) into groups: [0:2], [2:4], [4:6]
    j_groups = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    k_groups = [(0, 2), (2, 4), (4, 6)]  # sizes 2, 2, 2
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for jg_idx, (j_start, j_end) in enumerate(j_groups):
        j_size = j_end - j_start
        for kg_idx, (k_start, k_end) in enumerate(k_groups):
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen 2x2x2: rank 7
                strassen_decomp = [
                    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
                    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}),
                    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1}),
                    ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1}),
                    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),
                ]
                
                for U_dict, V_dict, W_dict in strassen_decomp:
                    U_row = {}
                    for (i, j_off), val in U_dict.items():
                        j = j_start + j_off
                        U_row[(i, j)] = val
                    V_row = {}
                    for (j_off, k_off), val in V_dict.items():
                        j = j_start + j_off
                        k = k_start + k_off
                        V_row[(j, k)] = val
                    W_row = {}
                    for (k_off, i), val in W_dict.items():
                        k = k_start + k_off
                        W_row[(k, i)] = val
                    
                    U_terms.append(U_row)
                    V_terms.append(V_row)
                    W_terms.append(W_row)
                    
            elif j_size == 2 and k_size == 1:
                # 2x2x1: schoolbook, rank 4
                for i in range(2):
                    for j_off in range(2):
                        j = j_start + j_off
                        k = k_start
                        U_terms.append({(i, j): 1})
                        V_terms.append({(j, k): 1})
                        W_terms.append({(k, i): 1})
                        
            elif j_size == 1 and k_size == 2:
                # 2x1x2: schoolbook, rank 4
                for i in range(2):
                    for k_off in range(2):
                        j = j_start
                        k = k_start + k_off
                        U_terms.append({(i, j): 1})
                        V_terms.append({(j, k): 1})
                        W_terms.append({(k, i): 1})
                        
            elif j_size == 1 and k_size == 1:
                # 2x1x1: schoolbook, rank 2
                for i in range(2):
                    j = j_start
                    k = k_start
                    U_terms.append({(i, j): 1})
                    V_terms.append({(j, k): 1})
                    W_terms.append({(k, i): 1})
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for (i, j), val in U_terms[r].items():
            U[i * m + j][r] = val
        for (j, k), val in V_terms[r].items():
            V[j * p + k][r] = val
        for (k, i), val in W_terms[r].items():
            W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
