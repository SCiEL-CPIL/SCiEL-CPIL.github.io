import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Block decomposition with block size 2
    block_size = 2
    n_blocks = n // block_size  # 3

    # Total rank is n_blocks^3 * 7
    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    # W index is k*n + i, where k is col of C, i is row of C
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen definitions for 2x2 block matrices:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    # Post-processing: find and remove linearly dependent columns
    # to reduce rank while preserving exact decomposition
    if time.time() < deadline:
        try:
            U_float = U.astype(float)
            V_float = V.astype(float)
            W_float = W.astype(float)

            # Find pivot columns in U (rows are independent)
            # We need to find the maximum set of columns that are linearly independent
            # Use QR with column pivoting
            Q, R, piv = np.linalg.qr(U_float, mode='complete', pivoting=True)
            # R is upper triangular, rank = number of non-zero diagonal entries
            tol = 1e-10
            diag = np.abs(np.diag(R))
            rank_u = np.sum(diag > tol)

            # Similarly for V
            Q_v, R_v, piv_v = np.linalg.qr(V_float, mode='complete', pivoting=True)
            diag_v = np.abs(np.diag(R_v))
            rank_v = np.sum(diag_v > tol)

            # Similarly for W
            Q_w, R_w, piv_w = np.linalg.qr(W_float, mode='complete', pivoting=True)
            diag_w = np.abs(np.diag(R_w))
            rank_w = np.sum(diag_w > tol)

            # The effective rank is the minimum of the three
            effective_rank = min(rank_u, rank_v, rank_w)
            current_rank = U.shape[1]

            if effective_rank < current_rank:
                # Keep only the pivot columns
                U_kept = U[:, piv[:effective_rank]]
                V_kept = V[:, piv[:effective_rank]]
                W_kept = W[:, piv[:effective_rank]]

                # Verify the decomposition is still exact
                # Check: sum_k U[:,k] * V[:,k] * W[:,k] == multiplication tensor
                # We'll do a spot check with random matrices
                np.random.seed(42)
                A = np.random.randint(-2, 3, (n, m))
                B = np.random.randint(-2, 3, (m, p))
                C_expected = A @ B

                # Compute using decomposition
                C_decomp = np.zeros((n, p))
                for k in range(effective_rank):
                    u_col = U_kept[:, k]
                    v_col = V_kept[:, k]
                    w_col = W_kept[:, k]
                    # Reshape to matrices
                    U_mat = u_col.reshape(n, m)
                    V_mat = v_col.reshape(m, p)
                    W_mat = w_col.reshape(p, n)
                    # C[i,j] += sum_k U[i,k'] * V[k',j'] * W[j,i']
                    # Actually the decomposition is: C[i,j] = sum_k U[i*m+j, k] * V[j*p+i, k] * W[i*n+j, k]
                    # Wait, let me re-read the convention
                    # U has shape (n*m, rank), indexed by (i, k) where i is row of A, k is col of A
                    # V has shape (m*p, rank), indexed by (k, j) where k is row of B, j is col of B
                    # W has shape (p*n, rank), indexed by (j, i) where j is col of C, i is row of C
                    # C[i,j] = sum_k U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
                    pass

                # Simpler verification: check if the tensors match
                # The tensor T[i,j,k] = delta[i*m+k, j*p+k] ... no
                # T[i,j,k] = 1 if i*m+k == row_U and k*p+j == row_V and j*n+i == row_W
                # Actually the standard form: C = A * B means C[i,j] = sum_k A[i,k] * B[k,j]
                # Tensor form: T[i,j,k] = A[i,k] * B[k,j] * C[i,j]
                # The decomposition says: T = sum_t U_t ⊗ V_t ⊗ W_t
                # where U_t[i*m+k] = u_ik, V_t[k*p+j] = v_kj, W_t[j*n+i] = w_ji
                # So T[i,j,k] = sum_t u_ik * v_kj * w_ji

                # Let's verify by checking a few entries
                verified = True
                for i in range(min(n, 3)):
                    for j in range(min(p, 3)):
                        for k in range(min(m, 3)):
                            # Expected: 1 (since T[i,j,k] = 1 for all valid i,j,k)
                            # Actually no, the multiplication tensor has T[i,j,k] = 1
                            # because C[i,j] = sum_k A[i,k]*B[k,j]
                            # So the tensor is all 1s in the valid range
                            expected_val = 1.0
                            decomp_val = 0.0
                            for t in range(effective_rank):
                                u_val = U_kept[i * m + k, t]
                                v_val = V_kept[k * p + j, t]
                                w_val = W_kept[j * n + i, t]
                                decomp_val += u_val * v_val * w_val
                            if abs(decomp_val - expected_val) > 1e-8:
                                verified = False
                                break
                        if not verified:
                            break
                    if not verified:
                        break

                if verified:
                    U = U_kept
                    V = V_kept
                    W = W_kept
        except Exception:
            pass  # Keep original if verification fails

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
