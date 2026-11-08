import numpy as np
from itertools import product
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Rank 98 decomposition: two independent 4x4x4 multiplications
    # Each 4x4x4 uses recursive Strassen (rank 49)
    
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def get_strassen_2x2x2_terms():
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
    
    strassen_2x2x2 = get_strassen_2x2x2_terms()
    
    # Define the 7 block products for 4x4x4
    block_products = [
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
    ]
    
    col = 0
    for half in range(2): # 0: left, 1: right
        b_offset = half * 4
        
        for bp_idx in range(7):
            A_blocks, B_blocks, C_blocks = block_products[bp_idx]
            
            for s_idx in range(7):
                u_pat, v_pat, w_pat = strassen_2x2x2[s_idx]
                
                u_vec = np.zeros(16)
                v_vec = np.zeros(32)
                w_vec = np.zeros(32)
                
                for (ar, ac, a_coeff) in A_blocks:
                    for (lr, ls, u_coeff) in u_pat.items():
                        global_i = ar * 2 + lr
                        global_k = ac * 2 + ls
                        global_A_idx = global_i * 4 + global_k
                        u_vec[global_A_idx] += a_coeff * u_coeff
                
                for (br, bc, b_coeff) in B_blocks:
                    for (lr, ls, v_coeff) in v_pat.items():
                        global_k = br * 2 + lr
                        local_j = bc * 2 + ls
                        global_j = b_offset + local_j
                        global_B_idx = global_k * 8 + global_j
                        v_vec[global_B_idx] += b_coeff * v_coeff
                
                for (cr, cc, c_coeff) in C_blocks:
                    for (lr, ls, w_coeff) in w_pat.items():
                        global_i = cr * 2 + lr
                        local_j = cc * 2 + ls
                        global_j = b_offset + local_j
                        global_W_idx = global_j * 4 + global_i
                        w_vec[global_W_idx] += c_coeff * w_coeff
                
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
