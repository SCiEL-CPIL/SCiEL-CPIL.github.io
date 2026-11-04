import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    
    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_strassen_2x2x2(i_off, j_off, k_off):
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1], [j0*p+k0, j1*p+k1], [1, 1], [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1], [j0*p+k0], [1], [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term([i0*m+j0], [1], [j0*p+k1, j1*p+k1], [1, -1], [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j1], [1], [j1*p+k0, j0*p+k0], [1, -1], [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1], [j1*p+k1], [1], [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1], [j0*p+k0, j0*p+k1], [1, 1], [k1*n+i1], [1])
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1], [j1*p+k0, j1*p+k1], [1, 1], [k0*n+i0], [1])

    def gen_schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii, jj, kk = i_off + i, j_off + j, k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    def gen_2x2x3(i_off, j_off, k_off):
        """2x2 * 2x3: Strassen(7) + schoolbook for last col(4) = 11"""
        gen_strassen_2x2x2(i_off, j_off, k_off)
        for i in range(2):
            for j in range(2):
                ii, jj = i_off + i, j_off + j
                kk = k_off + 2
                add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    def gen_2x2x4(i_off, j_off, k_off):
        """2x2 * 2x4: two Strassen(7+7) = 14"""
        gen_strassen_2x2x2(i_off, j_off, k_off)
        gen_strassen_2x2x2(i_off, j_off, k_off + 2)

    def gen_2x2x7(i_off, j_off, k_off):
        """2x2 * 2x7: 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1 = 7+7+7+4 = 25"""
        gen_strassen_2x2x2(i_off, j_off, k_off)
        gen_strassen_2x2x2(i_off, j_off, k_off + 2)
        gen_strassen_2x2x2(i_off, j_off, k_off + 4)
        # 2x2x1 schoolbook
        for i in range(2):
            for j in range(2):
                ii, jj = i_off + i, j_off + j
                kk = k_off + 6
                add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    # Build: split n=3 as 2+1, m=4 as 2+2
    # 2x4x7: two 2x2x7 blocks = 2*25 = 50
    gen_2x2x7(0, 0, 0)
    gen_2x2x7(0, 2, 0)
    # 1x4x7: schoolbook = 28
    gen_schoolbook(1, 4, 7, 2, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    # Verify
    ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    U_sel = U[ii * m + jj]
    V_sel = V[jj * p + kk]
    W_sel = W[kk * n + ii]
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if not np.allclose(vals, 1.0, atol=1e-9):
        # Fallback to schoolbook
        U_vecs_fb, V_vecs_fb, W_vecs_fb = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i*m+j] = 1.0; v[j*p+k] = 1.0; w[k*n+i] = 1.0
                    U_vecs_fb.append(u); V_vecs_fb.append(v); W_vecs_fb.append(w)
        U = np.array(U_vecs_fb).T
        V = np.array(V_vecs_fb).T
        W = np.array(W_vecs_fb).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
