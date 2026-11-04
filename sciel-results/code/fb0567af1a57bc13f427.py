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
    
    # Block partition: [2, 1, 2]
    n_parts = [2, 1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    
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
            
    def add_strassen_2x2x2(i_start, j_start, k_start):
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        add_term({i0*m+j0: 1, i1*m+j1: 1},
               {j0*p+k0: 1, j1*p+k1: 1},
               {k0*n+i0: 1, k1*n+i1: 1})
               
        add_term({i1*m+j0: 1, i1*m+j1: 1},
               {j0*p+k0: 1},
               {k0*n+i1: 1, k1*n+i1: -1})
               
        add_term({i0*m+j0: 1},
               {j0*p+k1: 1, j1*p+k1: -1},
               {k1*n+i0: 1, k1*n+i1: 1})
               
        add_term({i1*m+j1: 1},
               {j1*p+k0: 1, j0*p+k0: -1},
               {k0*n+i0: 1, k0*n+i1: 1})
               
        add_term({i0*m+j0: 1, i0*m+j1: 1},
               {j1*p+k1: 1},
               {k0*n+i0: -1, k1*n+i0: 1})
               
        add_term({i1*m+j0: 1, i0*m+j0: -1},
               {j0*p+k0: 1, j0*p+k1: 1},
               {k1*n+i1: 1})
               
        add_term({i0*m+j1: 1, i1*m+j1: -1},
               {j1*p+k0: 1, j1*p+k1: 1},
               {k0*n+i0: 1})

    def add_block(ni, mj, pk, i_start, j_start, k_start):
        if ni == 2 and mj == 2 and pk == 2:
            add_strassen_2x2x2(i_start, j_start, k_start)
        elif ni == 2 and mj == 1 and pk == 2:
            # Outer product: Rank 1
            # C[i, k] = A[i, 0] * B[0, k]
            u_d = {}
            v_d = {}
            w_d = {}
            for i in range(ni):
                for k in range(pk):
                    u_d[(i_start+i)*m + (j_start)] = 1
                    v_d[(j_start)*p + (k_start+k)] = 1
                    w_d[(k_start+k)*n + (i_start+i)] = 1
            add_term(u_d, v_d, w_d)
        elif ni == 1 and mj == 1 and pk == 2:
            # Outer product: Rank 1
            # C[0, k] = A[0, 0] * B[0, k]
            u_d = {}
            v_d = {}
            w_d = {}
            for k in range(pk):
                u_d[(i_start)*m + (j_start)] = 1
                v_d[(j_start)*p + (k_start+k)] = 1
                w_d[(k_start+k)*n + (i_start)] = 1
            add_term(u_d, v_d, w_d)
        elif ni == 2 and mj == 1 and pk == 1:
            # Outer product: Rank 1
            # C[i, 0] = A[i, 0] * B[0, 0]
            u_d = {}
            v_d = {}
            w_d = {}
            for i in range(ni):
                u_d[(i_start+i)*m + (j_start)] = 1
                v_d[(j_start)*p + (k_start)] = 1
                w_d[(k_start)*n + (i_start+i)] = 1
            add_term(u_d, v_d, w_d)
        elif ni == 1 and mj == 1 and pk == 1:
            # Scalar: Rank 1
            u_d = {(i_start)*m + (j_start): 1}
            v_d = {(j_start)*p + (k_start): 1}
            w_d = {(k_start)*n + (i_start): 1}
            add_term(u_d, v_d, w_d)
        elif ni == 1 and mj == 2 and pk == 1:
            # Dot product: Rank 2
            # C[0, 0] = A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0]
            add_term({(i_start)*m + (j_start): 1},
                   {(j_start)*p + (k_start): 1},
                   {(k_start)*n + (i_start): 1})
            add_term({(i_start)*m + (j_start+1): 1},
                   {(j_start+1)*p + (k_start): 1},
                   {(k_start)*n + (i_start): 1})
        elif ni == 2 and mj == 2 and pk == 1:
            # Matrix-Vector: Rank 4
            # C[i, 0] = A[i, 0]*B[0, 0] + A[i, 1]*B[1, 0]
            for i in range(ni):
                for j in range(mj):
                    add_term({(i_start+i)*m + (j_start+j): 1},
                           {(j_start+j)*p + (k_start): 1},
                           {(k_start)*n + (i_start+i): 1})
        elif ni == 1 and mj == 2 and pk == 2:
            # Matrix-Vector: Rank 4
            # C[0, k] = A[0, 0]*B[0, k] + A[0, 1]*B[1, k]
            for j in range(mj):
                for k in range(pk):
                    add_term({(i_start)*m + (j_start+j): 1},
                           {(j_start+j)*p + (k_start+k): 1},
                           {(k_start+k)*n + (i_start): 1})
        else:
            # Fallback schoolbook
            for u in range(ni):
                for v in range(mj):
                    for w in range(pk):
                        add_term({(i_start+u)*m + (j_start+v): 1},
                               {(j_start+v)*p + (k_start+w): 1},
                               {(k_start+w)*n + (i_start+u): 1})

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                add_block(ni, mj, pk, i_start, j_start, k_start)
    
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
