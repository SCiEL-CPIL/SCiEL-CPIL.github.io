import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get linear form index
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(j, k):
        return j * p + k
        
    def c_idx(i, k):
        return k * n + i

    # Define outer Strassen steps
    # Each step r defines P_r and Q_r as 2x2 matrices of linear forms
    # P_r[i][j] is a list of (coeff, a_index)
    # Q_r[i][j] is a list of (coeff, b_index)
    
    steps = []
    
    # M1 = (A00 + A11)(B00 + B11)
    P1 = [
        [(1, a_idx(0,0)), (1, a_idx(2,2))], [(1, a_idx(0,1)), (1, a_idx(2,3))],
        [(1, a_idx(1,0)), (1, a_idx(3,2))], [(1, a_idx(1,1)), (1, a_idx(3,3))]
    ]
    Q1 = [
        [(1, b_idx(0,0)), (1, b_idx(2,2))], [(1, b_idx(0,1)), (1, b_idx(2,3))],
        [(1, b_idx(1,0)), (1, b_idx(3,2))], [(1, b_idx(1,1)), (1, b_idx(3,3))]
    ]
    steps.append((P1, Q1, [(0,0,1), (1,1,1), (0,1,-1), (1,0,1)])) # C00, C11, C01? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The contribution of M_r to C blocks:
    # M1: C00(1), C01(0), C10(0), C11(1) -> Wait, M1 is a 2x2 matrix.
    # M1[0,0] -> C[0,0]. M1[0,1] -> C[0,1]. M1[1,0] -> C[1,0]. M1[1,1] -> C[1,1].
    # So M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coeff 1.
    # But C00 is a block. C00 = M1 + ...
    # So M1[0,0] adds to C[0,0].
    # M1[0,1] adds to C[0,1].
    # M1[1,0] adds to C[1,0].
    # M1[1,1] adds to C[1,1].
    
    # So for each M_r, we have a 2x2 matrix of coefficients for C entries.
    # M1: [[1, 0], [0, 1]]? No.
    # C00 is the top-left block.
    # C00 = M1 + M4 - M5 + M7.
    # This means the entire 2x2 matrix M1 is added to C[0:2, 0:2].
    # So M1[0,0] -> C[0,0]. M1[0,1] -> C[0,1]. M1[1,0] -> C[1,0]. M1[1,1] -> C[1,1].
    # All with coeff 1.
    
    # M2: C10 = M2 + M4.
    # So M2 is added to C[2:4, 0:2].
    # M2[0,0] -> C[2,0]. M2[0,1] -> C[2,1]. M2[1,0] -> C[3,0]. M2[1,1] -> C[3,1].
    # Coeff 1.
    
    # M3: C01 = M3 + M5.
    # M3 added to C[0:2, 2:4].
    # M3[0,0] -> C[0,2]. M3[0,1] -> C[0,3]. M3[1,0] -> C[1,2]. M3[1,1] -> C[1,3].
    # Coeff 1.
    
    # M4: C00 (1), C10 (1).
    # M4 added to C[0:2, 0:2] and C[2:4, 0:2].
    # M4[0,0] -> C[0,0] (1), C[2,0] (1).
    # M4[0,1] -> C[0,1] (1), C[2,1] (1).
    # M4[1,0] -> C[1,0] (1), C[3,0] (1).
    # M4[1,1] -> C[1,1] (1), C[3,1] (1).
    
    # M5: C00 (-1), C01 (1).
    # M5 added to C[0:2, 0:2] (-1) and C[0:2, 2:4] (1).
    
    # M6: C11 (1).
    # M6 added to C[2:4, 2:4].
    
    # M7: C00 (1).
    # M7 added to C[0:2, 0:2].
    
    # Let's define the mapping for each M_r to C entries.
    # mapping[r][i][j] = list of (c_i, c_k, coeff)
    
    mapping = {}
    # M1
    mapping[1] = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)]] # This is wrong structure.
    # Let's use a dict: mapping[r][i][j] = { (c_i, c_k): coeff }
    
    def add_map(r, pi, pj, ci, ck, coeff):
        if r not in mapping: mapping[r] = {}
        if pi not in mapping[r]: mapping[r][pi] = {}
        if pj not in mapping[r][pi]: mapping[r][pi][pj] = {}
        mapping[r][pi][pj][(ci, ck)] = coeff

    # M1
    for i in range(2):
        for j in range(2):
            add_map(1, i, j, i, j, 1)
    # M2
    for i in range(2):
        for j in range(2):
            add_map(2, i, j, i+2, j, 1)
    # M3
    for i in range(2):
        for j in range(2):
            add_map(3, i, j, i, j+2, 1)
    # M4
    for i in range(2):
        for j in range(2):
            add_map(4, i, j, i, j, 1)
            add_map(4, i, j, i+2, j, 1)
    # M5
    for i in range(2):
        for j in range(2):
            add_map(5, i, j, i, j, -1)
            add_map(5, i, j, i, j+2, 1)
    # M6
    for i in range(2):
        for j in range(2):
            add_map(6, i, j, i+2, j+2, 1)
    # M7
    for i in range(2):
        for j in range(2):
            add_map(7, i, j, i, j, 1)

    # Now define P and Q for each step
    # P_r[i][j] is a dict of a_index -> coeff
    # Q_r[i][j] is a dict of b_index -> coeff
    
    def make_PQ(block_sum_A, block_sum_B):
        # block_sum_A is a list of (block, coeff)
        # block is (bi, bj)
        P = [[{} for _ in range(2)] for _ in range(2)]
        Q = [[{} for _ in range(2)] for _ in range(2)]
        
        for (bi, bj), coeff in block_sum_A:
            for i in range(2):
                for j in range(2):
                    ai = bi * 2 + i
                    aj = bj * 2 + j
                    idx = a_idx(ai, aj)
                    P[i][j][idx] = P[i][j].get(idx, 0) + coeff
        
        for (bi, bj), coeff in block_sum_B:
            for i in range(2):
                for j in range(2):
                    bi_ = bi * 2 + i
                    bj_ = bj * 2 + j
                    idx = b_idx(bi_, bj_)
                    Q[i][j][idx] = Q[i][j].get(idx, 0) + coeff
        return P, Q

    P1, Q1 = make_PQ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)])
    P2, Q2 = make_PQ([((1,0),1), ((1,1),1)], [((0,0),1)])
    P3, Q3 = make_PQ([((0,0),1)], [((0,1),1), ((1,1),-1)])
    P4, Q4 = make_PQ([((1,1),1)], [((1,0),1), ((0,0),-1)])
    P5, Q5 = make_PQ([((0,0),1), ((0,1),1)], [((1,1),1)])
    P6, Q6 = make_PQ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)])
    P7, Q7 = make_PQ([((1,1),1), ((0,1),-1)], [((1,0),1), ((1,1),1)])

    steps_PQ = [
        (1, P1, Q1),
        (2, P2, Q2),
        (3, P3, Q3),
        (4, P4, Q4),
        (5, P5, Q5),
        (6, P6, Q6),
        (7, P7, Q7)
    ]

    col = 0
    for r, P, Q in steps_PQ:
        # Inner Strassen
        # S1 = (P00 + P11)(Q00 + Q11)
        # S2 = (P10 + P11)Q00
        # S3 = P00(Q01 - Q11)
        # S4 = P11(Q10 - Q00)
        # S5 = (P00 + P01)Q11
        # S6 = (P10 - P00)(Q00 + Q01)
        # S7 = (P11 - P01)(Q10 + Q11)
        
        def combine(P1, P2, sign2=1):
            res = {}
            for idx, c in P1.items(): res[idx] = res.get(idx, 0) + c
            for idx, c in P2.items(): res[idx] = res.get(idx, 0) + sign2 * c
            return res
            
        def combine_Q(Q1, Q2, sign2=1):
            res = {}
            for idx, c in Q1.items(): res[idx] = res.get(idx, 0) + c
            for idx, c in Q2.items(): res[idx] = res.get(idx, 0) + sign2 * c
            return res

        S1_A = combine(P[0][0], P[1][1])
        S1_B = combine_Q(Q[0][0], Q[1][1])
        # S1 contributes to M[0][0] (1) and M[1][1] (1)
        # M[0][0] -> C[i,j] via mapping
        # M[1][1] -> C[i,j] via mapping
        
        # We need to add the rank 1 term for S1
        # U: S1_A, V: S1_B
        # W: sum of contributions
        
        w_vec = {}
        for (pi, pj), contrib in mapping[r].items():
             # Wait, mapping[r] is { pi: { pj: { (ci, ck): coeff } } }
             # S1 contributes to M[0][0] and M[1][1] with coeff 1.
             pass
        
        # Let's explicitly list the contributions of Sk to M_r[i][j]
        # M[0][0] = S1 + S4 - S5 + S7
        # M[0][1] = S3 + S5
        # M[1][0] = S2 + S4
        # M[1][1] = S1 - S2 + S3 + S6
        
        contribs = [
            (S1_A, S1_B, { (0,0): 1, (1,1): 1 }),
            (combine(P[1][0], P[1][1]), Q[0][0], { (1,0): 1 }),
            (P[0][0], combine_Q(Q[0][1], Q[1][1], -1), { (0,1): 1 }),
            (P[1][1], combine_Q(Q[1][0], Q[0][0], -1), { (0,0): 1, (1,0): 1 }),
            (combine(P[0][0], P[0][1]), Q[1][1], { (0,1): 1, (0,0): -1 }),
            (combine(P[1][0], P[0][0], -1), combine_Q(Q[0][0], Q[0][1]), { (1,1): 1 }),
            (combine(P[1][1], P[0][1], -1), combine_Q(Q[1][0], Q[1][1]), { (0,0): 1, (1,1): 1 })
        ]
        
        for LA, LB, m_contrib in contribs:
            # LA, LB are dicts of index -> coeff
            # m_contrib is dict (pi, pj) -> coeff
            w_coeffs = {}
            for (pi, pj), coeff_m in m_contrib.items():
                for (ci, ck), coeff_c in mapping[r][pi][pj].items():
                    total = coeff_m * coeff_c
                    idx_w = c_idx(ci, ck)
                    w_coeffs[idx_w] = w_coeffs.get(idx_w, 0) + total
            
            # Fill U, V, W
            for idx_a, c in LA.items():
                U[idx_a, col] = c
            for idx_b, c in LB.items():
                V[idx_b, col] = c
            for idx_w, c in w_coeffs.items():
                W[idx_w, col] = c
            col += 1
            
    return {"U": U, "V": V, "W": W}
