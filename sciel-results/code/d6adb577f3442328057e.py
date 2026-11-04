import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS

    def build_strassen_decomp():
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)

            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val

            U_vecs.append(u)
            V_vecs.append(v)
            W_vecs.append(w)

        def strassen_222_local(i0, j0, k0):
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

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

        def schoolbook_block(i0, j0, k0, ni, mj, pk):
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        ii = i0 + i
                        jj = j0 + j
                        kk = k0 + k
                        add_rank1_term([ii*m+jj], [1.0],
                                       [jj*p+kk], [1.0],
                                       [kk*n+ii], [1.0])

        def block_decomp(i0, j0, k0, ni, mj, pk):
            if ni == 2 and mj == 2 and pk == 2:
                strassen_222_local(i0, j0, k0)
            else:
                schoolbook_block(i0, j0, k0, ni, mj, pk)

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(0, j0, k0, 2, 2, 2)
        for j0 in [0, 2]:
            block_decomp(0, j0, 6, 2, 2, 1)
        
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(2, j0, k0, 1, 2, 2)
        for j0 in [0, 2]:
            block_decomp(2, j0, 6, 1, 2, 1)
        
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    U, V, W = build_strassen_decomp()
    rank = U.shape[1]

    # Vectorize the tensors
    # T is n x m x p
    # Flatten to n*m*p
    # U is n*m x rank
    # V is m*p x rank
    # W is p*n x rank
    
    # We want to find the rank of the set of vectors U_r * V_r * W_r
    # This is not simply the rank of the concatenated matrix.
    # The tensor T = sum R_r (u_r v_r^T w_r^T)
    # We can form a matrix M where column r is the vectorization of T_r.
    
    # T_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's build M
    # Size n*m*p x rank
    M = np.zeros((n*m*p, rank))
    for r in range(rank):
        # Build T_r
        # i, j, k
        # U_idx = i*m+j
        # V_idx = j*p+k
        # W_idx = k*n+i
        
        # We can do this efficiently
        # Reshape U[:, r] to n x m
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        
        # T_r[i, j, k] = u_r[i, j] * v_r[j, k] * w_r[k, i]
        # This is a bit tricky to vectorize directly without loops.
        # But n, m, p are small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i*m*p + j*p + k, r] = u_r[i, j] * v_r[j, k] * w_r[k, i]
                    
    # Now find the rank of M
    # Use SVD
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = 1e-10
    num_pivots = np.sum(s > tol)
    
    # Select the first num_pivots columns of M?
    # No, we need to select the corresponding columns of U, V, W.
    # The SVD gives us a basis for the column space of M.
    # The columns of M are linearly dependent.
    # We can just take the first num_pivots columns?
    # Not necessarily. We need a subset of the original columns that is linearly independent.
    # QR decomposition with column pivoting is better.
    
    Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
    # piv is the permutation of columns of M.T (rows of M)
    # Wait, QR of M.T means we are finding a basis for the row space of M.
    # We want a basis for the column space of M.
    # So we should do QR on M.
    
    Q, R, piv = np.linalg.qr(M, mode='complete', pivoting=True)
    # R is upper triangular.
    # The first num_pivots rows of R are non-zero.
    # The corresponding columns in M (permuted by piv) are linearly independent.
    
    # piv is the permutation of columns of M.
    # So the independent columns are M[:, piv[:num_pivots]]
    
    # Let's verify
    M_indep = M[:, piv[:num_pivots]]
    rank_check = np.linalg.matrix_rank(M_indep, tol=tol)
    if rank_check < num_pivots:
         # Adjust num_pivots
         num_pivots = rank_check
         
    # Select the corresponding U, V, W columns
    U_final = U[:, piv[:num_pivots]]
    V_final = V[:, piv[:num_pivots]]
    W_final = W[:, piv[:num_pivots]]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
