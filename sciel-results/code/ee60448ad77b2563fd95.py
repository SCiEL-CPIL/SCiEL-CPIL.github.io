"""Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with optimized sub-block decompositions.
    """
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
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

    def schoolbook_block(i0, j0, k0, ni, mj, pk):
        """Schoolbook decomposition for an ni x mj x pk block."""
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    def m223_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x3 block.
        Uses Strassen for the 2x2x2 part + schoolbook for 2x2x1 = rank 11.
        """
        strassen_222(i0, j0, k0)
        schoolbook_block(i0, j0, k0 + 2, 2, 2, 1)

    def m224_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x4 block.
        Uses two Strassen 2x2x2 blocks = rank 14.
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        schoolbook_block(i0, j0, k0 + 6, 2, 2, 1)

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook = rank 14.
        """
        schoolbook_block(i0, j0, k0, 1, 2, 7)

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25
          M_{1,2,7}: 14
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2:
          M_{3,2,7}: 39
          M_{3,2,7}: 39
        Total: 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the full decomposition
    if time.time() - start_time < deadline:
        m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness with a few random tests
    if time.time() - start_time < deadline - 5.0:
        for trial in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B

            # Efficient verification using matrix operations
            # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
            
            # Reshape for efficient computation
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            W_mat = W.reshape(p, n, rank)
            
            # For each r, compute contribution
            C_computed = np.zeros((n, p))
            for r in range(rank):
                U_r = U_mat[:, :, r]  # (n, m)
                V_r = V_mat[:, :, r]  # (m, p)
                W_r = W_mat[:, :, r]  # (p, n)
                # C += W_r.T * (A @ U_r) * (V_r @ B.T).T
                # Actually: C[i,k] += W_r[k,i] * sum_j A[i,j]*U_r[i,j] * sum_j B[j,k]*V_r[j,k]
                # Let's compute: for each r, the contribution is
                # C += np.einsum('ik,im,jk->ij', W_r, A*U_r, B*V_r) -- no, that's not right
                # C[i,k] += W_r[k,i] * (A @ U_r)[i,:] @ ... no
                # Let me think again:
                # C[i,k] = sum_r W_r[k,i] * (sum_j A[i,j] * U_r[i,j]) * (sum_j B[j,k] * V_r[j,k])
                # = sum_r W_r[k,i] * (A * U_r).sum(axis=1)[i] * (V_r * B).sum(axis=0)[k]
                
                A_times_Ur = (A * U_r).sum(axis=1)  # (n,)
                V_r_times_B = (V_r * B).sum(axis=0)  # (p,)
                C_computed += np.outer(W_r.T, A_times_Ur) * V_r_times_B  # Wait, this isn't right either
                
                # Actually: C[i,k] += W_r[k,i] * A_times_Ur[i] * V_r_times_B[k]
                # = outer(W_r.T, A_times_Ur) * V_r_times_B
                # where W_r.T has shape (n, p) with W_r.T[i,k] = W_r[k,i]
                pass
            
            # Let me just do it the direct way but vectorized
            C_computed = np.zeros((n, p))
            for r in range(rank):
                U_r = U_mat[:, :, r]  # (n, m)
                V_r = V_mat[:, :, r]  # (m, p)
                W_r = W_mat[:, :, r]  # (p, n)
                
                # a[i] = sum_j A[i,j] * U_r[i,j]
                a = (A * U_r).sum(axis=1)  # (n,)
                # b[k] = sum_j B[j,k] * V_r[j,k]
                b = (V_r * B).sum(axis=0)  # (p,)
                # C[i,k] += W_r[k,i] * a[i] * b[k]
                C_computed += W_r.T * np.outer(a, b)
            
            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-9:
                # Fallback to schoolbook
                rank_school = n * m * p
                U_sb = [[0] * rank_school for _ in range(n * m)]
                V_sb = [[0] * rank_school for _ in range(m * p)]
                W_sb = [[0] * rank_school for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i * m + j][col] = 1
                            V_sb[j * p + k][col] = 1
                            W_sb[k * n + i][col] = 1
                            col += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
