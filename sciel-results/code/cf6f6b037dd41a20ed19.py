import time
import numpy as np

def construct():
    start_time = time.time()
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

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off):
        """Schoolbook for a block."""
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen x3 + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        # Schoolbook for k = k0+6
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        
        Strategy: Split n=3 into 2+1.
        For the 2x4 part: split m=4 into 2+2.
          M_{2,2,7} + M_{2,2,7} = 25 + 25 = 50
        For the 1x4 part: schoolbook = 28
        Total: 78
        """
        # 2x4x7 part (i0, i0+1)
        m227_decomp(i0, j0, k0)
        m227_decomp(i0, j0 + 2, k0)
        # 1x4x7 part (i0+2)
        schoolbook_block(1, 4, 7, i0 + 2, j0, k0)

    # Build the full decomposition
    m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness
    deadline = 140.0  # safety margin before 150s kill
    if time.time() - start_time < deadline - 5.0:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Efficient verification using matrix products
        # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This can be computed as: for each r, compute outer products
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r]  # (n*m,)
            v_r = V[:, r]  # (m*p,)
            w_r = W[:, r]  # (p*n,)
            
            # Reshape to matrices
            u_mat = u_r.reshape(n, m)  # (n, m)
            v_mat = v_r.reshape(m, p)  # (m, p)
            w_mat = w_r.reshape(p, n)  # (p, n)
            
            # The contribution to C from rank r is:
            # C_r[i,k] = sum_j A[i,j] * u_mat[i,j] * (sum_j' ... ) 
            # Actually, the tensor decomposition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # This is not a simple matrix product. Let me compute it directly but efficiently.
            # For each i, k:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r] * B[j,k] * V[j*p+k, r])
            
            # Let me define for each r:
            # M_r[i,j] = A[i,j] * U[i*m+j, r]
            # N_r[j,k] = B[j,k] * V[j*p+k, r]
            # Then C[i,k] += W[k*n+i, r] * sum_j M_r[i,j] * N_r[j,k]
            # = W[k*n+i, r] * (M_r @ N_r)[i,k]
            
            M_r = A * u_mat
            N_r = B * v_mat
            C_computed += w_mat.T * (M_r @ N_r)

        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-9:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_sb = np.zeros((n * m, rank_school))
            V_sb = np.zeros((m * p, rank_school))
            W_sb = np.zeros((p * n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
