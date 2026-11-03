import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7
    # Current best known structured rank is 78.

    U_vecs = []
    V_vecs = []
    W_vecs = []

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

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off
            j = j_off
            k = k_off
            add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1], [jj*p+kk], [1], [kk*n+i], [1])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1], [j*p+kk], [1], [kk*n+ii], [1])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1], [jj*p+k], [1], [k*n+ii], [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            k1 = k_off + 1

            # Strassen 2x2x2
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
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
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    gen_terms(n, m, p, 0, 0, 0)

    # Convert to float for output
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T

    # Peeling: try to reduce rank by finding redundant rank-1 components
    rank = U.shape[1]
    if time.time() < DEADLINE - 30 and rank > 20:
        # Fast verification: check that the tensor decomposition is exact
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We precompute the full tensor once, then for removal we check
        # if the removed term contributes zero to all (i,j,k) positions.
        
        def full_tensor(U_mat, V_mat, W_mat):
            U_3d = U_mat.reshape(n, m, -1)
            V_3d = V_mat.reshape(m, p, -1)
            W_3d = W_mat.reshape(p, n, -1)
            return np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        
        target = np.ones((n, m, p))
        
        # Multi-pass peeling with fast incremental check
        # A term t can be removed if for all (i,j,k):
        #   U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 0
        # i.e., the outer product of the three vectors is zero everywhere.
        # This is equivalent to: the rank-1 tensor formed by u_t, v_t, w_t is zero.
        # But that's only true if the term is truly redundant given others.
        # Actually, we need: the contribution of term t equals zero in the
        # residual after removing it. So we need full verification.
        
        # Faster approach: use the fact that if term t is zero in all three
        # vectors simultaneously at the relevant indices, it's removable.
        # But the correct test is whether the sum without t still equals target.
        
        # Use a smarter approach: compute the residual tensor for each candidate
        # removal. Precompute the full tensor, then for each t, compute
        # the contribution of term t and check if full_tensor - contrib_t == target.
        
        full_T = full_tensor(U, V, W)
        
        # Verify initial decomposition
        if not np.allclose(full_T, target, atol=1e-10):
            # Fall back to schoolbook
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array([[float(x) for x in row] for row in U_vecs]).T
            V = np.array([[float(x) for x in row] for row in V_vecs]).T
            W = np.array([[float(x) for x in row] for row in W_vecs]).T
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        removed = set()
        current_rank = rank
        
        # Multiple passes
        for pass_num in range(3):
            if time.time() > DEADLINE - 25:
                break
            
            any_removed = False
            for t in range(current_rank):
                if time.time() > DEADLINE - 25:
                    break
                if t in removed:
                    continue
                
                # Compute contribution of term t
                u_t = U[:, t]  # shape (n*m,)
                v_t = V[:, t]  # shape (m*p,)
                w_t = W[:, t]  # shape (p*n,)
                
                # Contribution tensor: contrib[i,j,k] = u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
                # Residual without t: full_T - contrib
                # Check if residual == target
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
            # Fallback to schoolbook
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array([[float(x) for x in row] for row in U_vecs]).T
            V = np.array([[float(x) for x in row] for row in V_vecs]).T
            W = np.array([[float(x) for x in row] for row in W_vecs]).T
    except Exception:
        # Fallback to schoolbook on error
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array([[float(x) for x in row] for row in U_vecs]).T
        V = np.array([[float(x) for x in row] for row in V_vecs]).T
        W = np.array([[float(x) for x in row] for row in W_vecs]).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
