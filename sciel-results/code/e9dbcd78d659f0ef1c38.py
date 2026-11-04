import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7

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

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        # Base cases and specific optimizations
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+i], [1.0])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1.0], [j*p+kk], [1.0], [kk*n+ii], [1.0])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+k], [1.0], [k*n+ii], [1.0])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            # Strassen 2x2x2
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off; k1 = k_off+1
            
            # M1 = (A11+A22)(B11+B22)
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            # M2 = (A21+A22)B11
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            # M3 = A11(B21-B22)
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            # M4 = A22(B12-B11)
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            # M5 = (A11+A12)B22
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            # M6 = (A21-A11)(B11+B12)
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
            # M7 = (A12-A22)(B21+B22)
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])
            return

        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return

        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return

        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return

        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    gen_terms(n, m, p, 0, 0, 0)

    # Convert to arrays
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    rank = U.shape[1]

    # Peeling: try to reduce rank by finding redundant rank-1 components
    if time.time() < DEADLINE - 30 and rank > 20:
        def full_tensor(U_mat, V_mat, W_mat):
            U_3d = U_mat.reshape(n, m, -1)
            V_3d = V_mat.reshape(m, p, -1)
            W_3d = W_mat.reshape(p, n, -1)
            return np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        
        target = np.ones((n, m, p))
        
        full_T = full_tensor(U, V, W)
        
        if not np.allclose(full_T, target, atol=1e-10):
            # Fallback to schoolbook if something went wrong
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            U = np.array(U_vecs).T
            V = np.array(V_vecs).T
            W = np.array(W_vecs).T
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        removed = set()
        current_rank = rank
        
        # Multiple passes
        for pass_num in range(10):
            if time.time() > DEADLINE - 25:
                break
            
            any_removed = False
            for t in range(current_rank):
                if time.time() > DEADLINE - 25:
                    break
                if t in removed:
                    continue
                
                u_t = U[:, t]
                v_t = V[:, t]
                w_t = W[:, t]
                
                contrib = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib[i, j, k] = u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
                
                residual = full_T - contrib
                if np.allclose(residual, target, atol=1e-10):
                    removed.add(t)
                    any_removed = True
            
            if not any_removed:
                break
        
        if removed:
            final_mask = [i for i in range(rank) if i not in removed]
            U = U[:, final_mask]
            V = V[:, final_mask]
            W = W[:, final_mask]
            rank = U.shape[1]

    # Final verification
    try:
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=1e-10):
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            U = np.array(U_vecs).T
            V = np.array(V_vecs).T
            W = np.array(W_vecs).T
    except Exception:
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
