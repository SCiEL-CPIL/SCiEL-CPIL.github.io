"""Matrix multiplication 5x5x5 with Strassen-based decomposition."""
import time

DIMENSIONS = [5, 5, 5]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def strassen_term():
        terms = []
        terms.append(([ (0,1),(3,1) ], [ (0,1),(3,1) ], [ (0,1),(3,1) ]))
        terms.append(([ (2,1),(3,1) ], [ (0,1) ], [ (2,1) ]))
        terms.append(([ (0,1) ], [ (1,1),(3,-1) ], [ (1,1),(3,-1) ]))
        terms.append(([ (3,1) ], [ (2,1),(0,-1) ], [ (0,1),(2,1) ]))
        terms.append(([ (0,1),(1,1) ], [ (3,1) ], [ (0,-1),(1,1) ]))
        terms.append(([ (2,1),(0,-1) ], [ (0,1),(2,1) ], [ (3,1) ]))
        terms.append(([ (1,1),(3,-1) ], [ (1,1),(3,1) ], [ (0,1),(1,-1) ]))
        return terms
    
    for r1 in range(3):
        for r2 in range(3):
            for c1 in range(3):
                s_r1, s_r2, s_c1 = sizes[r1], sizes[r2], sizes[c1]
                
                if s_r1 == 2 and s_r2 == 2 and s_c1 == 2:
                    for u_e, v_e, w_e in strassen_term():
                        u_full = []
                        for (li, val) in u_e:
                            i_g = offsets[r1] + li // 2
                            j_g = offsets[r2] + li % 2
                            u_full.append((i_g * m + j_g, val))
                        v_full = []
                        for (lj, val) in v_e:
                            j_g = offsets[r2] + lj // 2
                            k_g = offsets[c1] + lj % 2
                            v_full.append((j_g * p + k_g, val))
                        w_full = []
                        for (lk, val) in w_e:
                            k_g = offsets[c1] + lk // 2
                            i_g = offsets[r1] + lk % 2
                            w_full.append((k_g * n + i_g, val))
                        U_terms.append(u_full)
                        V_terms.append(v_full)
                        W_terms.append(w_full)
                else:
                    for i_l in range(s_r1):
                        for j_l in range(s_r2):
                            for k_l in range(s_c1):
                                i_g = offsets[r1] + i_l
                                j_g = offsets[r2] + j_l
                                k_g = offsets[c1] + k_l
                                U_terms.append([(i_g * m + j_g, 1)])
                                V_terms.append([(j_g * p + k_g, 1)])
                                W_terms.append([(k_g * n + i_g, 1)])
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row, val in U_terms[col]:
            U[row][col] = val
        for row, val in V_terms[col]:
            V[row][col] = val
        for row, val in W_terms[col]:
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
