import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
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

    def strassen_2x2x2(i_off, j_off, k_off):
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1

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

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    add_rank1_term([i_off*m+(j_off+j)], [1.0],
                                   [(j_off+j)*p+(k_off+k)], [1.0],
                                   [(k_off+k)*n+i_off], [1.0])
            return
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    add_rank1_term([(i_off+i)*m+j_off], [1.0],
                                   [j_off*p+(k_off+k)], [1.0],
                                   [(k_off+k)*n+(i_off+i)], [1.0])
            return
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    add_rank1_term([(i_off+i)*m+(j_off+j)], [1.0],
                                   [(j_off+j)*p+k_off], [1.0],
                                   [k_off*n+(i_off+i)], [1.0])
            return
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            return
        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off+2, j_off, k_off)
            return
        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off+2, k_off)
            return
        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off+2)
            gen_terms(2, 2, 2, i_off, j_off, k_off+4)
            gen_terms(2, 2, 1, i_off, j_off, k_off+6)
            return
        # Fallback: schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    add_rank1_term([(i_off+i)*m+(j_off+j)], [1.0],
                                   [(j_off+j)*p+(k_off+k)], [1.0],
                                   [(k_off+k)*n+(i_off+i)], [1.0])

    gen_terms(n, m, p, 0, 0, 0)

    U = np.array(U_vecs).T  # (rank, n*m)
    V = np.array(V_vecs).T  # (rank, m*p)
    W = np.array(W_vecs).T  # (rank, p*n)
    R = U.shape[0]

    # Verification helper using einsum
    def is_valid(Um, Vm, Wm):
        U_t = Um.reshape(R, n, m).transpose(1, 2, 0)  # (n, m, R)
        V_t = Vm.reshape(R, m, p).transpose(1, 2, 0)  # (m, p, R)
        W_t = Wm.reshape(R, p, n).transpose(1, 2, 0)  # (p, n, R)
        Q = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.allclose(Q, 1.0, atol=1e-9)

    # Confirm base decomposition is valid
    assert is_valid(U, V, W), "Base decomposition failed verification"

    # Local-search refinement: try to eliminate redundant terms
    # A term is redundant if removing it and adjusting others still satisfies the tensor.
    # Simple heuristic: try removing each term; if the resulting tensor has all entries
    # still close to 1 after allowing small corrections, keep the removal.
    # More practical: try removing terms where the contribution is "cancellable".

    # Strategy: for each term, check if its contribution is linearly dependent on others.
    # Build the contribution matrix: each term c contributes to T[i,j,k] via
    #   U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i]
    # If we remove term c, the residual is the negative of its contribution.
    # We can try to see if the residual can be absorbed by scaling other terms,
    # but that's complex. Instead, try a simple greedy removal:
    #   For each term in reverse order, try removing it. If the tensor is still valid
    #   (within tolerance), keep it removed.

    # This is expensive to check naively, so we limit the number of attempts.
    max_removal_attempts = 30
    removed = 0
    for attempt in range(max_removal_attempts):
        if time.time() > deadline:
            break
        # Try removing the last remaining term (or a random one)
        if R - removed <= 0:
            break
        idx_to_try = R - removed - 1  # Try from the end
        U_trial = np.delete(U, idx_to_try, axis=0)
        V_trial = np.delete(V, idx_to_try, axis=0)
        W_trial = np.delete(W, idx_to_try, axis=0)

        # Quick check: compute the tensor with the trial matrices
        R_trial = U_trial.shape[0]
        U_t = U_trial.reshape(R_trial, n, m).transpose(1, 2, 0)
        V_t = V_trial.reshape(R_trial, m, p).transpose(1, 2, 0)
        W_t = W_trial.reshape(R_trial, p, n).transpose(1, 2, 0)
        Q = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)

        if np.allclose(Q, 1.0, atol=1e-9):
            U = U_trial
            V = V_trial
            W = W_trial
            R = R_trial
            removed += 1
        else:
            # Try a few other indices before giving up
            found_removal = False
            for alt in range(R - removed - 2, -1, -1):
                if alt < 0:
                    break
                U_trial = np.delete(U, alt, axis=0)
                V_trial = np.delete(V, alt, axis=0)
                W_trial = np.delete(W, alt, axis=0)
                R_trial = U_trial.shape[0]
                U_t = U_trial.reshape(R_trial, n, m).transpose(1, 2, 0)
                V_t = V_trial.reshape(R_trial, m, p).transpose(1, 2, 0)
                W_t = W_trial.reshape(R_trial, p, n).transpose(1, 2, 0)
                Q = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                if np.allclose(Q, 1.0, atol=1e-9):
                    U = U_trial
                    V = V_trial
                    W = W_trial
                    R = R_trial
                    removed += 1
                    found_removal = True
                    break
            if not found_removal:
                break  # No more easy removals

    # Final verification
    if R > 0:
        U_t = U.reshape(R, n, m).transpose(1, 2, 0)
        V_t = V.reshape(R, m, p).transpose(1, 2, 0)
        W_t = W.reshape(R, p, n).transpose(1, 2, 0)
        Q = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        if not np.allclose(Q, 1.0, atol=1e-9):
            # Fallback to original if refinement broke something
            U_vecs2 = []
            V_vecs2 = []
            W_vecs2 = []
            # Rebuild original
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            gen_terms(n, m, p, 0, 0, 0)
            U = np.array(U_vecs).T
            V = np.array(V_vecs).T
            W = np.array(W_vecs).T
            R = U.shape[0]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
