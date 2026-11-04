import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a rank-reduced tensor decomposition for 3x5 * 5x5 matrix multiplication
    using block decomposition with Strassen's algorithm for 2x2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Partition indices into blocks
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Strassen decomposition for 2x2x2
                    strassen_terms = [
                        {
                            'U': {(0,0): 1, (1,1): 1},
                            'V': {(0,0): 1, (1,1): 1},
                            'W': {(0,0): 1, (1,1): 1}
                        },
                        {
                            'U': {(1,0): 1, (1,1): 1},
                            'V': {(0,0): 1},
                            'W': {(0,1): 1, (1,1): -1}
                        },
                        {
                            'U': {(0,0): 1},
                            'V': {(0,1): 1, (1,1): -1},
                            'W': {(1,0): 1, (1,1): 1}
                        },
                        {
                            'U': {(1,1): 1},
                            'V': {(1,0): 1, (0,0): -1},
                            'W': {(0,0): 1, (0,1): 1}
                        },
                        {
                            'U': {(0,0): 1, (0,1): 1},
                            'V': {(1,1): 1},
                            'W': {(0,0): -1, (1,0): 1}
                        },
                        {
                            'U': {(1,0): 1, (0,0): -1},
                            'V': {(0,0): 1, (0,1): 1},
                            'W': {(1,1): 1}
                        },
                        {
                            'U': {(0,1): 1, (1,1): -1},
                            'V': {(1,0): 1, (1,1): 1},
                            'W': {(0,0): 1}
                        },
                    ]
                    
                    for term in strassen_terms:
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        
                        for (li, lj), coeff in term['U'].items():
                            gi = i_b[li]
                            gj = j_b[lj]
                            u_vec[gi * m + gj] = coeff
                        
                        for (lj, lk), coeff in term['V'].items():
                            gj = j_b[lj]
                            gk = k_b[lk]
                            v_vec[gj * p + gk] = coeff
                        
                        for (lk, li), coeff in term['W'].items():
                            gk = k_b[lk]
                            gi = i_b[li]
                            w_vec[gk * n + gi] = coeff
                        
                        U_cols.append(u_vec)
                        V_cols.append(v_vec)
                        W_cols.append(w_vec)
                        
                        if time.time() > deadline:
                            return None
                else:
                    # Schoolbook for this sub-block
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                u_vec = [0] * (n * m)
                                v_vec = [0] * (m * p)
                                w_vec = [0] * (p * n)
                                
                                gi = i_b[li]
                                gj = j_b[lj]
                                gk = k_b[lk]
                                
                                u_vec[gi * m + gj] = 1
                                v_vec[gj * p + gk] = 1
                                w_vec[gk * n + gi] = 1
                                
                                U_cols.append(u_vec)
                                V_cols.append(v_vec)
                                W_cols.append(w_vec)
                                
                                if time.time() > deadline:
                                    return None
    
    rank = len(U_cols)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
