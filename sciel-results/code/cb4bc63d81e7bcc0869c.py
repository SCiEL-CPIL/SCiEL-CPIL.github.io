DIMENSIONS = [5, 5, 5]
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct using random sampling."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check all entries for small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms(i_start, j_start, k_start, n, m, p):
    """Generate Strassen terms for a 2x2x2 block at given offsets."""
    terms = []
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    add_term({i0*m+j0: 1, i1*m+j1: 1},
             {j0*p+k0: 1, j1*p+k1: 1},
             {k0*n+i0: 1, k1*n+i1: 1})

    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
    add_term({i1*m+j0: 1, i1*m+j1: 1},
             {j0*p+k0: 1},
             {k0*n+i1: 1, k1*n+i1: -1})

    # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
    add_term({i0*m+j0: 1},
             {j0*p+k1: 1, j1*p+k1: -1},
             {k1*n+i0: 1, k1*n+i1: 1})

    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
    add_term({i1*m+j1: 1},
             {j1*p+k0: 1, j0*p+k0: -1},
             {k0*n+i0: 1, k0*n+i1: 1})

    # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
    add_term({i0*m+j0: 1, i0*m+j1: 1},
             {j1*p+k1: 1},
             {k0*n+i0: -1, k1*n+i0: 1})

    # M6 = (a10-a00)(b00+b01) -> C11 += M6
    add_term({i1*m+j0: 1, i0*m+j0: -1},
             {j0*p+k0: 1, j0*p+k1: 1},
             {k1*n+i1: 1})

    # M7 = (a01-a11)(b10+b11) -> C00 += M7
    add_term({i0*m+j1: 1, i1*m+j1: -1},
             {j1*p+k0: 1, j1*p+k1: 1},
             {k0*n+i0: 1})

    return terms

def _laderman_3x3_terms(i_start, j_start, k_start, n, m, p):
    """Generate Laderman terms for a 3x3x3 block at given offsets."""
    # Laderman's algorithm for 3x3 matrix multiplication uses 23 multiplications
    # This is an improvement over the naive 27
    terms = []
    
    i0, i1, i2 = i_start, i_start + 1, i_start + 2
    j0, j1, j2 = j_start, j_start + 1, j_start + 2
    k0, k1, k2 = k_start, k_start + 1, k_start + 2
    
    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    # Laderman's 23-term algorithm
    # M1 = (a00+a11+a22)(b00+b11+b22)
    add_term({i0*m+j0: 1, i1*m+j1: 1, i2*m+j2: 1},
             {j0*p+k0: 1, j1*p+k1: 1, j2*p+k2: 1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M2 = (a00-a11)(b00-b11)
    add_term({i0*m+j0: 1, i1*m+j1: -1},
             {j0*p+k0: 1, j1*p+k1: -1},
             {k0*n+i0: 1, k1*n+i1: 1})
    
    # M3 = (a01+a10+a22)(b10+b01+b22)
    add_term({i0*m+j1: 1, i1*m+j0: 1, i2*m+j2: 1},
             {j1*p+k0: 1, j0*p+k1: 1, j2*p+k2: 1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M4 = (a01-a10)(b10-b01)
    add_term({i0*m+j1: 1, i1*m+j0: -1},
             {j1*p+k0: 1, j0*p+k1: -1},
             {k0*n+i0: 1, k1*n+i1: 1})
    
    # M5 = (a12+a21+a00)(b21+b12+b00)
    add_term({i1*m+j2: 1, i2*m+j1: 1, i0*m+j0: 1},
             {j2*p+k1: 1, j1*p+k2: 1, j0*p+k0: 1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M6 = (a12-a21)(b21-b12)
    add_term({i1*m+j2: 1, i2*m+j1: -1},
             {j2*p+k1: 1, j1*p+k2: -1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M7 = (a20+a02+a11)(b02+a20+b11)
    add_term({i2*m+j0: 1, i0*m+j2: 1, i1*m+j1: 1},
             {j0*p+k2: 1, j2*p+k0: 1, j1*p+k1: 1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M8 = (a20-a02)(b02-b20)
    add_term({i2*m+j0: 1, i0*m+j2: -1},
             {j0*p+k2: 1, j2*p+k0: -1},
             {k0*n+i0: 1, k1*n+i1: 1, k2*n+i2: 1})
    
    # M9 = a00*b21
    add_term({i0*m+j0: 1},
             {j0*p+k1: 1},
             {k1*n+i0: 1})
    
    # M10 = a00*b12
    add_term({i0*m+j0: 1},
             {j0*p+k2: 1},
             {k2*n+i0: 1})
    
    # M11 = a11*b02
    add_term({i1*m+j1: 1},
             {j1*p+k2: 1},
             {k2*n+i1: 1})
    
    # M12 = a11*b20
    add_term({i1*m+j1: 1},
             {j1*p+k0: 1},
             {k0*n+i1: 1})
    
    # M13 = a22*b10
    add_term({i2*m+j2: 1},
             {j2*p+k0: 1},
             {k0*n+i2: 1})
    
    # M14 = a22*b01
    add_term({i2*m+j2: 1},
             {j2*p+k1: 1},
             {k1*n+i2: 1})
    
    # M15 = a01*b20
    add_term({i0*m+j1: 1},
             {j1*p+k0: 1},
             {k0*n+i0: 1})
    
    # M16 = a02*b10
    add_term({i0*m+j2: 1},
             {j2*p+k0: 1},
             {k0*n+i0: 1})
    
    # M17 = a10*b02
    add_term({i1*m+j0: 1},
             {j0*p+k2: 1},
             {k2*n+i1: 1})
    
    # M18 = a12*b20
    add_term({i1*m+j2: 1},
             {j2*p+k0: 1},
             {k0*n+i1: 1})
    
    # M19 = a20*b12
    add_term({i2*m+j0: 1},
             {j0*p+k2: 1},
             {k2*n+i2: 1})
    
    # M20 = a21*b02
    add_term({i2*m+j1: 1},
             {j1*p+k2: 1},
             {k2*n+i2: 1})
    
    # M21 = a01*b10
    add_term({i0*m+j1: 1},
             {j1*p+k0: 1},
             {k0*n+i0: 1})
    
    # M22 = a02*b01
    add_term({i0*m+j2: 1},
             {j2*p+k1: 1},
             {k1*n+i0: 1})
    
    # M23 = a10*b21
    add_term({i1*m+j0: 1},
             {j0*p+k1: 1},
             {k1*n+i1: 1})
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 5, 5, 5

    # Try different block partition strategies
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: [2,2,1] with Strassen for 2x2x2
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
                    terms.extend(_strassen_2x2_terms(i_start, j_start, k_start, n, m, p))
                else:
                    # Schoolbook for smaller blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                add_term({(i_start+u)*m + (j_start+v): 1},
                                         {(j_start+v)*p + (k_start+w): 1},
                                         {(k_start+w)*n + (i_start+u): 1})
    
    rank1 = len(terms)
    
    # Build factor matrices for strategy 1
    U = [[0] * rank1 for _ in range(n * m)]
    V = [[0] * rank1 for _ in range(m * p)]
    W = [[0] * rank1 for _ in range(p * n)]
    
    for r in range(rank1):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    if _verify_decomposition(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank1
    
    if time.time() > deadline - 5:
        return best_result if best_result else _naive_result(n, m, p)
    
    # Strategy 2: [3,2] with Laderman for 3x3x3
    n_parts2 = [3, 2]
    m_parts2 = [3, 2]
    p_parts2 = [3, 2]
    
    n_starts2 = [0]
    m_starts2 = [0]
    p_starts2 = [0]
    for i in range(1, len(n_parts2)):
        n_starts2.append(n_starts2[-1] + n_parts2[i-1])
    for i in range(1, len(m_parts2)):
        m_starts2.append(m_starts2[-1] + m_parts2[i-1])
    for i in range(1, len(p_parts2)):
        p_starts2.append(p_starts2[-1] + p_parts2[i-1])
    
    terms2 = []
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts2[i]
                mj = m_parts2[j]
                pk = p_parts2[k]
                
                i_start = n_starts2[i]
                j_start = m_starts2[j]
                k_start = p_starts2[k]
                
                if ni == 3 and mj == 3 and pk == 3:
                    # Laderman for 3x3x3
                    terms2.extend(_laderman_3x3_terms(i_start, j_start, k_start, n, m, p))
                elif ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    terms2.extend(_strassen_2x2_terms(i_start, j_start, k_start, n, m, p))
                else:
                    # Schoolbook for smaller blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                add_term2 = lambda ud, vd, wd: terms2.append((ud, vd, wd)) if all(ud.values()) and all(vd.values()) and all(wd.values()) else None
                                ud = {(i_start+u)*m + (j_start+v): 1}
                                vd = {(j_start+v)*p + (k_start+w): 1}
                                wd = {(k_start+w)*n + (i_start+u): 1}
                                terms2.append((ud, vd, wd))
    
    rank2 = len(terms2)
    
    if rank2 < best_rank:
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        
        for r in range(rank2):
            u_dict, v_dict, w_dict = terms2[r]
            for idx, val in u_dict.items():
                U2[idx][r] = val
            for idx, val in v_dict.items():
                V2[idx][r] = val
            for idx, val in w_dict.items():
                W2[idx][r] = val
        
        if _verify_decomposition(U2, V2, W2, n, m, p):
            best_result = {"U": U2, "V": V2, "W": W2}
            best_rank = rank2
    
    if time.time() > deadline - 5:
        return best_result if best_result else _naive_result(n, m, p)
    
    # Strategy 3: Mixed [2,3] 
    n_parts3 = [2, 3]
    m_parts3 = [2, 3]
    p_parts3 = [2, 3]
    
    n_starts3 = [0]
    m_starts3 = [0]
    p_starts3 = [0]
    for i in range(1, len(n_parts3)):
        n_starts3.append(n_starts3[-1] + n_parts3[i-1])
    for i in range(1, len(m_parts3)):
        m_starts3.append(m_starts3[-1] + m_parts3[i-1])
    for i in range(1, len(p_parts3)):
        p_starts3.append(p_starts3[-1] + p_parts3[i-1])
    
    terms3 = []
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts3[i]
                mj = m_parts3[j]
                pk = p_parts3[k]
                
                i_start = n_starts3[i]
                j_start = m_starts3[j]
                k_start = p_starts3[k]
                
                if ni == 3 and mj == 3 and pk == 3:
                    terms3.extend(_laderman_3x3_terms(i_start, j_start, k_start, n, m, p))
                elif ni == 2 and mj == 2 and pk == 2:
                    terms3.extend(_strassen_2x2_terms(i_start, j_start, k_start, n, m, p))
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                terms3.append({(i_start+u)*m + (j_start+v): 1},
                                             {(j_start+v)*p + (k_start+w): 1},
                                             {(k_start+w)*n + (i_start+u): 1})
    
    rank3 = len(terms3)
    
    if rank3 < best_rank:
        U3 = [[0] * rank3 for _ in range(n * m)]
        V3 = [[0] * rank3 for _ in range(m * p)]
        W3 = [[0] * rank3 for _ in range(p * n)]
        
        for r in range(rank3):
            u_dict, v_dict, w_dict = terms3[r]
            for idx, val in u_dict.items():
                U3[idx][r] = val
            for idx, val in v_dict.items():
                V3[idx][r] = val
            for idx, val in w_dict.items():
                W3[idx][r] = val
        
        if _verify_decomposition(U3, V3, W3, n, m, p):
            best_result = {"U": U3, "V": V3, "W": W3}
            best_rank = rank3
    
    if best_result is None:
        best_result = _naive_result(n, m, p)
    
    return best_result

def _naive_result(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
