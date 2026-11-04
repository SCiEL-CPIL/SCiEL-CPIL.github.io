import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen steps: (L_coeffs, R_coeffs, W_coeffs)
    # Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1)
    strassen_2x2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
        ([(0,1)],         [(1,1),(3,-1)],[(1,1),(3,1)]),
        ([(3,1)],         [(2,1),(0,-1)],[(0,1),(2,1)]),
        ([(0,1),(1,1)],   [(3,1)],       [(0,-1),(1,1)]),
        ([(2,1),(0,-1)],  [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)],  [(2,1),(3,1)], [(0,1)]),
    ]
    
    # 4x4 Strassen steps: (L_blocks, R_blocks, c_contribs)
    # Block idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_4x4 = [
        ([(0,1),(3,1)],  [(0,1),(3,1)],  [(0,1),(3,1)]),
        ([(2,1),(3,1)],  [(0,1)],        [(2,1),(3,-1)]),
        ([(0,1)],        [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)],        [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)],  [(3,1)],        [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)],  [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)],  [(0,1)]),
    ]
    
    col_idx = 0
    for half in range(2):
        b_off = half * 4
        c_off = half * 4
        
        for k in range(7):
            L_blocks = strassen_4x4[k][0]
            R_blocks = strassen_4x4[k][1]
            c_contribs = strassen_4x4[k][2]
            
            for sub_k in range(7):
                L_sub = strassen_2x2[sub_k][0]
                R_sub = strassen_2x2[sub_k][1]
                W_sub = strassen_2x2[sub_k][2]
                
                # U contributions
                for (ei, ce) in L_sub:
                    er, ec = ei // 2, ei % 2
                    for (bi, cb) in L_blocks:
                        br, bc = bi // 2, bi % 2
                        a_r = 2 * br + er
                        a_c = 2 * bc + ec
                        U[a_r * 4 + a_c][col_idx] += cb * ce
                
                # V contributions
                for (ei, ce) in R_sub:
                    er, ec = ei // 2, ei % 2
                    for (bi, cb) in R_blocks:
                        br, bc = bi // 2, bi % 2
                        b_r = 2 * br + er
                        b_c = 2 * bc + ec + b_off
                        V[b_r * 8 + b_c][col_idx] += cb * ce
                
                # W contributions
                for (ei, ce) in W_sub:
                    er, ec = ei // 2, ei % 2
                    for (ci, cc) in c_contribs:
                        cr, ccj = ci // 2, ci % 2
                        c_r = 2 * cr + er
                        c_c = 2 * ccj + ec + c_off
                        W[c_c * 4 + c_r][col_idx] += cc * ce
                
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
