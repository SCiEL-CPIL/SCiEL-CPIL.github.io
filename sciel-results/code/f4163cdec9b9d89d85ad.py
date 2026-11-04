import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Build U, V, W as dense matrices.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We don't know rank upfront, so we'll accumulate into lists of columns
    # or just use a list of dicts/tuples and convert at the end.
    # Let's use a list of (row, col, val) for each matrix and then convert.
    
    U_entries = []  # (row, rank_idx, val)
    V_entries = []  # (row, rank_idx, val)
    W_entries = []  # (row, rank_idx, val)
    
    rank = 0
    
    def add_term(u_local, v_local, w_local, i_off, j_off, k_off):
        nonlocal rank
        r = rank
        rank += 1
        for (li, lj, c) in u_local:
            gi = i_off + li
            gj = j_off + lj
            U_entries.append((gi * m + gj, r, c))
        for (lj, lk, c) in v_local:
            gj = j_off + lj
            gk = k_off + lk
            V_entries.append((gj * p + gk, r, c))
        for (lk, li, c) in w_local:
            gk = k_off + lk
            gi = i_off + li
            W_entries.append((gk * n + gi, r, c))
    
    def strassen_2x2x2(i_off, j_off, k_off):
        # Standard Strassen decomposition for 2x2x2
        # M = A * B
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # t1 = (a+d)(e+h)
        add_term(
            [(0,0,1), (1,1,1)],  # a+d
            [(0,0,1), (1,1,1)],  # e+h
            [(0,0,1), (0,1,1), (1,0,1), (1,1,1)],  # contributes to all 4
            i_off, j_off, k_off
        )
        # Wait, the W matrix in tensor decomposition for M[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # is not simply the output matrix.
        # The tensor is T[i,j,k] = delta[i,k] (for the multiplication tensor? No)
        # The multiplication tensor T_{ijk} has T_{ijk} = 1 if i,k are output indices and j is contraction?
        # Standard definition: T_{i,j,k} = 1 if (i,j) in U, (j,k) in V, (k,i) in W.
        # Actually, the tensor is M_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor is T_{i,j,k} where A is indexed by i,j and B by j,k.
        # Usually, the tensor is defined on indices (i, j, k) such that
        # A_{ij} B_{jk} -> M_{ik}.
        # The decomposition is M_{ik} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} ?
        # The task says: "output factor follows the official trace convention W[k*n+i,r]".
        # This implies W is indexed by (k, i) -> row k*n+i.
        # So the term is U[i,j,r] * V[j,k,r] * W[k,i,r].
        # This matches M_{ik} = sum_r ...
        
        # Let's re-verify the Strassen terms in this notation.
        # We want to compute M_{ik}.
        
        # Term 1: (a+d)(e+h)
        # U: a (0,0), d (1,1) -> [(0,0,1), (1,1,1)]
        # V: e (0,0), h (1,1) -> [(0,0,1), (1,1,1)]
        # W: contributes to M_{0,0}, M_{0,1}, M_{1,0}, M_{1,1} with +1?
        # M_{0,0} += ae + ah + de + dh ? No.
        # (a+d)(e+h) = ae + ah + de + dh.
        # M_{0,0} = ae + bg. M_{0,1} = af + bh. M_{1,0} = ce + dg. M_{1,1} = cf + dh.
        # So T1 = ae + ah + de + dh.
        # We need to combine T1..T7 to get the correct M.
        
        # Standard Strassen:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-d)(g+h)
        # P7 = (b-c)(e+f)
        
        # M00 = P1 + P4 - P5 + P7
        # M01 = P3 + P5
        # M10 = P2 + P4
        # M11 = P1 - P2 + P3 + P6
        
        # Let's map these to U, V, W.
        
        # P1:
        # U: a(0,0), d(1,1)
        # V: e(0,0), h(1,1)
        # W: M00(+1), M01(+1), M10(+1), M11(+1)
        # W indices: (k,i). M00 is k=0,i=0. M01 is k=1,i=0. M10 is k=0,i=1. M11 is k=1,i=1.
        add_term(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,0,1), (0,1,1), (1,1,1)],
            i_off, j_off, k_off
        )
        
        # P2: (c+d)e
        # U: c(1,0), d(1,1)
        # V: e(0,0)
        # W: M10(+1), M11(-1)
        add_term(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            i_off, j_off, k_off
        )
        
        # P3: a(f-h)
        # U: a(0,0)
        # V: f(0,1), -h(1,1)
        # W: M01(+1), M11(+1)
        add_term(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            i_off, j_off, k_off
        )
        
        # P4: d(g-e)
        # U: d(1,1)
        # V: g(1,0), -e(0,0)
        # W: M00(+1), M10(+1)
        add_term(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            i_off, j_off, k_off
        )
        
        # P5: (a+b)h
        # U: a(0,0), b(0,1)
        # V: h(1,1)
        # W: M00(-1), M01(+1)
        add_term(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (1,0,1)],
            i_off, j_off, k_off
        )
        
        # P6: (c-d)(g+h)
        # U: c(1,0), -d(1,1)
        # V: g(1,0), h(1,1)
        # W: M11(+1)
        add_term(
            [(1,0,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(1,1,1)],
            i_off, j_off, k_off
        )
        
        # P7: (b-c)(e+f)
        # U: b(0,1), -c(1,0)
        # V: e(0,0), f(0,1)
        # W: M00(+1), M10(-1)
        add_term(
            [(0,1,1), (1,0,-1)],
            [(0,0,1), (0,1,1)],
            [(0,0,1), (0,1,-1)],
            i_off, j_off, k_off
        )
        
    def decompose(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif m_sub > 2 and m_sub % 2 == 0:
            m_half = m_sub // 2
            decompose(n_sub, m_half, p_sub, i_off, j_off, k_off)
            decompose(n_sub, m_half, p_sub, i_off, j_off + m_half, k_off)
        elif p_sub > 2 and p_sub % 2 == 0:
            p_half = p_sub // 2
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off)
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off + p_half)
        elif n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        else:
            # Schoolbook for remaining small cases (e.g. 1x2x2, 1x1x2, etc.)
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term(
                            [(i, j, 1)],
                            [(j, k, 1)],
                            [(k, i, 1)],
                            i_off, j_off, k_off
                        )
                        
    decompose(n, m, p, 0, 0, 0)
    
    # Convert entries to matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for (row, r, val) in U_entries:
        U[row, r] = val
    for (row, r, val) in V_entries:
        V[row, r] = val
    for (row, r, val) in W_entries:
        W[row, r] = val
        
    return {"U": U, "V": V, "W": W}
