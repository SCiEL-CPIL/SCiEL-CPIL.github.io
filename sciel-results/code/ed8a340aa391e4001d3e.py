import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)
        return (u, v, w)

    def gen_2x2x2(i_off, j_off, k_off):
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1
        s = [
            ([i0*m+j0, i1*m+j1], [1, 1],
             [j0*p+k0, j1*p+k1], [1, 1],
             [k0*n+i0, k1*n+i1], [1, 1]),
            ([i1*m+j0, i1*m+j1], [1, 1],
             [j0*p+k0], [1],
             [k0*n+i1, k1*n+i1], [1, -1]),
            ([i0*m+j0], [1],
             [j0*p+k1, j1*p+k1], [1, -1],
             [k1*n+i0, k1*n+i1], [1, 1]),
            ([i1*m+j1], [1],
             [j1*p+k0, j0*p+k0], [1, -1],
             [k0*n+i0, k0*n+i1], [1, 1]),
            ([i0*m+j0, i0*m+j1], [1, 1],
             [j1*p+k1], [1],
             [k0*n+i0, k1*n+i0], [-1, 1]),
            ([i1*m+j0, i0*m+j0], [1, -1],
             [j0*p+k0, j0*p+k1], [1, 1],
             [k1*n+i1], [1]),
            ([i0*m+j1, i1*m+j1], [1, -1],
             [j1*p+k0, j1*p+k1], [1, 1],
             [k0*n+i0], [1]),
        ]
        return [add_rank1_term(ui, uv, vi, vv, wi, wv) for (ui, uv, vi, vv, wi, wv) in s]

    def gen_block(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate rank-1 terms for an n_l x m_l x p_l block."""
        terms = []
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            terms.append(add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1]))
            return terms
        
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            return gen_2x2x2(i_off, j_off, k_off)
        
        # Recursive split: split along the largest dimension
        if n_l >= 2:
            n1 = n_l // 2
            n2 = n_l - n1
            terms.extend(gen_block(n1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n2, m_l, p_l, i_off + n1, j_off, k_off))
            return terms
        
        if m_l >= 2:
            m1 = m_l // 2
            m2 = m_l - m1
            terms.extend(gen_block(n_l, m1, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m2, p_l, i_off, j_off + m1, k_off))
            return terms
        
        if p_l >= 2:
            p1 = p_l // 2
            p2 = p_l - p1
            terms.extend(gen_block(n_l, m_l, p1, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m_l, p2, i_off, j_off, k_off + p1))
            return terms
        
        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i; jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
        return terms

    def count_terms(n_l, m_l, p_l):
        """Count rank-1 terms for a block using recursive splitting."""
        if n_l == 1 and m_l == 1 and p_l == 1:
            return 1
        if n_l == 1:
            return m_l * p_l
        if m_l == 1:
            return n_l * p_l
        if p_l == 1:
            return n_l * m_l
        if n_l == 2 and m_l == 2 and p_l == 2:
            return 7
        
        # Try all possible splits
        best = float('inf')
        # Split n
        for n1 in range(1, n_l):
            n2 = n_l - n1
            c = count_terms(n1, m_l, p_l) + count_terms(n2, m_l, p_l)
            if c < best:
                best = c
        # Split m
        for m1 in range(1, m_l):
            m2 = m_l - m1
            c = count_terms(n_l, m1, p_l) + count_terms(n_l, m2, p_l)
            if c < best:
                best = c
        # Split p
        for p1 in range(1, p_l):
            p2 = p_l - p1
            c = count_terms(n_l, m_l, p1) + count_terms(n_l, m_l, p2)
            if c < best:
                best = c
        
        # Also consider schoolbook
        schoolbook = n_l * m_l * p_l
        if schoolbook < best:
            best = schoolbook
        
        return best

    # Find optimal split
    optimal_rank = count_terms(n, m, p)
    
    # Now generate the actual decomposition using the optimal split
    def gen_optimal(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate terms using the optimal split strategy."""
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            return [add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])]
        
        if n_l == 1:
            terms = []
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            terms = []
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            terms = []
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            return gen_2x2x2(i_off, j_off, k_off)
        
        # Find optimal split
        best_cost = float('inf')
        best_split = None
        
        # Try splitting n
        for n1 in range(1, n_l):
            n2 = n_l - n1
            c = count_terms(n1, m_l, p_l) + count_terms(n2, m_l, p_l)
            if c < best_cost:
                best_cost = c
                best_split = ('n', n1, n2)
        
        # Try splitting m
        for m1 in range(1, m_l):
            m2 = m_l - m1
            c = count_terms(n_l, m1, p_l) + count_terms(n_l, m2, p_l)
            if c < best_cost:
                best_cost = c
                best_split = ('m', m1, m2)
        
        # Try splitting p
        for p1 in range(1, p_l):
            p2 = p_l - p1
            c = count_terms(n_l, m_l, p1) + count_terms(n_l, m_l, p2)
            if c < best_cost:
                best_cost = c
                best_split = ('p', p1, p2)
        
        # Schoolbook fallback
        schoolbook_cost = n_l * m_l * p_l
        if schoolbook_cost < best_cost:
            terms = []
            for i in range(n_l):
                for j in range(m_l):
                    for k in range(p_l):
                        ii = i_off + i; jj = j_off + j; kk = k_off + k
                        terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        terms = []
        dim, s1, s2 = best_split
        if dim == 'n':
            terms.extend(gen_optimal(s1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_optimal(s2, m_l, p_l, i_off + s1, j_off, k_off))
        elif dim == 'm':
            terms.extend(gen_optimal(n_l, s1, p_l, i_off, j_off, k_off))
            terms.extend(gen_optimal(n_l, s2, p_l, i_off, j_off + s1, k_off))
        else:
            terms.extend(gen_optimal(n_l, m_l, s1, i_off, j_off, k_off))
            terms.extend(gen_optimal(n_l, m_l, s2, i_off, j_off, k_off + s1))
        
        return terms

    all_terms = gen_optimal(n, m, p, 0, 0, 0)
    
    U_vecs = [t[0] for t in all_terms]
    V_vecs = [t[1] for t in all_terms]
    W_vecs = [t[2] for t in all_terms]
    
    # Verify
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if prod_sum != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    term = add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
                    U_vecs.append(term[0])
                    V_vecs.append(term[1])
                    W_vecs.append(term[2])
    
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
