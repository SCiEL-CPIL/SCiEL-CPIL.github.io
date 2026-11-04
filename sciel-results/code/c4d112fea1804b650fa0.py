"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Partition:
    # n: top 2 rows (i=0,1), bottom 1 row (i=2)
    # m: 2 groups of 2 (j=0,1 and j=2,3)
    # p: 4 groups of 2 (k=0,1 / 2,3 / 4,5 / 6,7)
    
    # Strassen 2x2x2 factors (local indices)
    # Each factor: (list of (U_local_i, U_local_j, coeff)), (list of (V_local_j, V_local_k, coeff)), (list of (W_local_k, W_local_i, coeff))
    strassen_factors = [
        # Factor 0: P1 = A[0,0]*(B[0,0]-B[0,1])
        {
            'U': [(0, 0, 1)],
            'V': [(0, 0, 1), (0, 1, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        # Factor 1: P2 = (A[0,0]+A[0,1])*B[1,1]
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        # Factor 2: P3 = (A[1,0]+A[1,1])*B[0,0]
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # Factor 3: P4 = A[1,1]*(B[1,0]-B[0,0])
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        # Factor 4: P5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # Factor 5: P6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # Factor 6: P7 = (A[1,0]-A[0,0])*(B[0,1]+B[1,1])
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 1, 1), (1, 1, 1)],
            'W': [(1, 1, -1)]
        }
    ]
    
    # Schoolbook 1x2x2 factors (local indices)
    schoolbook_122 = [
        # (j=0,k=0)
        {'U': [(0, 0, 1)], 'V': [(0, 0, 1)], 'W': [(0, 0, 1)]},
        # (j=0,k=1)
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1)], 'W': [(1, 0, 1)]},
        # (j=1,k=0)
        {'U': [(0, 1, 1)], 'V': [(1, 0, 1)], 'W': [(0, 0, 1)]},
        # (j=1,k=1)
        {'U': [(0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(1, 0, 1)]},
    ]
    
    # Count total rank
    # 8 Strassen blocks (2 m-groups * 4 p-groups) * 7 = 56
    # 8 Schoolbook blocks (2 m-groups * 4 p-groups) * 4 = 32
    rank = 8 * 7 + 8 * 4
    # rank = 88
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # m-groups: (0,1) and (2,3)
    # p-groups: (0,1), (2,3), (4,5), (6,7)
    m_groups = [(0, 1), (2, 3)]
    p_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for j_off in m_groups:
        for k_off in p_groups:
            # Top 2 rows of A (i=0,1): Strassen 2x2x2
            for factor in strassen_factors:
                # U: local (i_l, j_l) -> global i=0+0=0+1? No:
                # i_local maps to i = i_off + i_l, where i_off=0 for top 2 rows
                # So global i = i_l (since i_off=0)
                # global j = j_off + j_l
                # global k = k_off + k_l
                for (i_l, j_l, coeff) in factor['U']:
                    global_i = i_l  # i_off = 0
                    global_j = j_off[0] + j_l
                    u_idx = global_i * m + global_j
                    U[u_idx][col] += coeff
                for (j_l, k_l, coeff) in factor['V']:
                    global_j = j_off[0] + j_l
                    global_k = k_off[0] + k_l
                    v_idx = global_j * p + global_k
                    V[v_idx][col] += coeff
                for (k_l, i_l, coeff) in factor['W']:
                    global_k = k_off[0] + k_l
                    global_i = i_l  # i_off = 0
                    w_idx = global_k * n + global_i
                    W[w_idx][col] += coeff
                col += 1
            
            # Bottom 1 row of A (i=2): Schoolbook 1x2x2
            for factor in schoolbook_122:
                for (i_l, j_l, coeff) in factor['U']:
                    global_i = 2  # bottom row
                    global_j = j_off[0] + j_l
                    u_idx = global_i * m + global_j
                    U[u_idx][col] += coeff
                for (j_l, k_l, coeff) in factor['V']:
                    global_j = j_off[0] + j_l
                    global_k = k_off[0] + k_l
                    v_idx = global_j * p + global_k
                    V[v_idx][col] += coeff
                for (k_l, i_l, coeff) in factor['W']:
                    global_k = k_off[0] + k_l
                    global_i = 2  # bottom row
                    w_idx = global_k * n + global_i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
