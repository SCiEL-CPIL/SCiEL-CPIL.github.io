import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145
    n, m, p = 3, 4, 7

    def check_deadline():
        if time.time() > DEADLINE:
            raise TimeoutError()

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        for i, val in zip(u_idx, u_val):
            u[i] += Fraction(val)
        for i, val in zip(v_idx, v_val):
            v[i] += Fraction(val)
        for i, val in zip(w_idx, w_val):
            w[i] += Fraction(val)
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_2x2x2(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        add_term([i0*m+j0, i1*m+j1], [1,1], [j0*p+k0, j1*p+k1], [1,1], [k0*n+i0, k1*n+i1], [1,1])
        add_term([i1*m+j0, i1*m+j1], [1,1], [j0*p+k0], [1], [k0*n+i1, k1*n+i1], [1,-1])
        add_term([i0*m+j0], [1], [j0*p+k1, j1*p+k1], [1,-1], [k1*n+i0, k1*n+i1], [1,1])
        add_term([i1*m+j1], [1], [j1*p+k0, j0*p+k0], [1,-1], [k0*n+i0, k0*n+i1], [1,1])
        add_term([i0*m+j0, i0*m+j1], [1,1], [j1*p+k1], [1], [k0*n+i0, k1*n+i0], [-1,1])
        add_term([i1*m+j0, i0*m+j0], [1,-1], [j0*p+k0, j0*p+k1], [1,1], [k1*n+i1], [1])
        add_term([i0*m+j1, i1*m+j1], [1,-1], [j1*p+k0, j1*p+k1], [1,1], [k0*n+i0], [1])

    def schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii, jj, kk = i_off+i, j_off+j, k_off+k
                    add_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    def gen(n_l, m_l, p_l, i_off, j_off, k_off):
        check_deadline()
        if n_l == 0 or m_l == 0 or p_l == 0:
            return
        if n_l == 1 and m_l == 1 and p_l == 1:
            add_term([i_off*m+j_off], [1], [j_off*p+k_off], [1], [k_off*n+i_off], [1])
        elif n_l == 2 and m_l == 2 and p_l == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif n_l == 1:
            schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off)
        elif m_l == 1:
            schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off)
        elif p_l == 1:
            schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off)
        else:
            # Recursive splitting
            if n_l >= 2:
                n1 = n_l // 2
                n2 = n_l - n1
                gen(n1, m_l, p_l, i_off, j_off, k_off)
                gen(n2, m_l, p_l, i_off + n1, j_off, k_off)
            elif m_l >= 2:
                m1 = m_l // 2
                m2 = m_l - m1
                gen(n_l, m1, p_l, i_off, j_off, k_off)
                gen(n_l, m2, p_l, i_off, j_off + m1, k_off)
            else:
                schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off)

    gen(n, m, p, 0, 0, 0)

    def to_array(vecs):
        arr = np.zeros((len(vecs[0]), len(vecs)), dtype=float)
        for r, vec in enumerate(vecs):
            for c, val in enumerate(vec):
                arr[c, r] = float(val)
        return arr

    U = to_array(U_vecs)
    V = to_array(V_vecs)
    W = to_array(W_vecs)

    def verify(U, V, W):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    if not verify(U, V, W):
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        schoolbook_block(n, m, p, 0, 0, 0)
        U = to_array(U_vecs)
        V = to_array(V_vecs)
        W = to_array(W_vecs)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
