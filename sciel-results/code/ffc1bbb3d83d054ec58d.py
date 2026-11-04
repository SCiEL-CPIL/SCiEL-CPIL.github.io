import time
_start_time = time.time()
_TIME_LIMIT = 110

DIMENSIONS = [5, 5, 5]

def construct():
    global _start_time, _TIME_LIMIT
    if time.time() - _start_time > _TIME_LIMIT:
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # Block partition: [2, 2, 1]
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1
                    
                    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                    add_term({i0*m+j0: 1, i1*m+j1: 1},
                           {j0*p+k0: 1, j1*p+k1: 1},
                           {k0*n+i0: 1, k1*n+i1: 1})
                    
                    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
                    add_term({i1*m+j0: 1, i1*m+j1: 1},
                           {j0*p+k0: 1},
                           {k0*n+i1: 1, k1*n+i1: -1})
                    
                    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
                    add_term({i0*m+j0: 1},
                           {j0*p+k1: 1, j1*p+k1: -1},
                           {k1*n+i0: 1, k1*n+i1: 1})
                    
                    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
                    add_term({i1*m+j1: 1},
                           {j1*p+k0: 1, j0*p+k0: -1},
                           {k0*n+i0: 1, k0*n+i1: 1})
                    
                    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
                    add_term({i0*m+j0: 1, i0*m+j1: 1},
                           {j1*p+k1: 1},
                           {k0*n+i0: -1, k1*n+i0: 1})
                    
                    # M6 = (A10-A00)(B00+B01) -> C11 += M6
                    add_term({i1*m+j0: 1, i0*m+j0: -1},
                           {j0*p+k0: 1, j0*p+k1: 1},
                           {k1*n+i1: 1})
                    
                    # M7 = (A01-A11)(B10+B11) -> C00 += M7
                    add_term({i0*m+j1: 1, i1*m+j1: -1},
                           {j1*p+k0: 1, j1*p+k1: 1},
                           {k0*n+i0: 1})
                else:
                    # Schoolbook for smaller blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                add_term({(i_start+u)*m + (j_start+v): 1},
                                       {(j_start+v)*p + (k_start+w): 1},
                                       {(k_start+w)*n + (i_start+u): 1})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
