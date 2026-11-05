import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS
    dim = n * m * p  # 56

    # Construct rank 50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)

    col = 0

    def emit_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M2
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        # M3
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M4
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + i0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        # M5
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        # M6
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M7
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    # 6 Strassen blocks
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 0, 4)
    emit_strassen(0, 2, 4)

    # 8 schoolbook terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
            col += 1

    # Verify
    U_3d = U.reshape(n, m, rank_init)
    V_3d = V.reshape(m, p, rank_init)
    W_3d = W.reshape(p, n, rank_init)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-9):
        return None

    current_rank = rank_init
    current_U, current_V, current_W = U, V, W

    def compute_tensor_matrix(U_mat, V_mat, W_mat, rank):
        """Compute the tensor matrix efficiently."""
        T_mat = np.zeros((dim, rank))
        for r in range(rank):
            U_r = U_mat[:, r].reshape(n, m)
            V_r = V_mat[:, r].reshape(m, p)
            W_r = W_mat[:, r].reshape(p, n)
            T_r = np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
            T_mat[:, r] = T_r.flatten()
        return T_mat

    def verify_decomposition(U_mat, V_mat, W_mat, rank, atol=1e-9):
        """Verify that the decomposition is correct."""
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=atol)

    def try_reduce_rank(current_U, current_V, current_W, current_rank, tol=1e-9, max_attempts=5):
        """Try to reduce rank by finding null vectors."""
        if time.time() > DEADLINE - 3:
            return current_U, current_V, current_W, current_rank, False

        T_mat = compute_tensor_matrix(current_U, current_V, current_W, current_rank)

        try:
            U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=False)
        except:
            return current_U, current_V, current_W, current_rank, False

        rank_mat = np.sum(s > tol)
        if rank_mat >= current_rank:
            return current_U, current_V, current_W, current_rank, False

        null_vecs = Vh[rank_mat:]

        # Try multiple null vectors
        for nv in null_vecs[:max_attempts]:
            mx = np.max(np.abs(nv))
            if mx < tol:
                continue
            k = np.argmax(np.abs(nv))
            c_k = nv[k]
            if abs(c_k) < tol:
                continue

            new_rank = current_rank - 1
            new_U = np.zeros((n*m, new_rank))
            new_V = np.zeros((m*p, new_rank))
            new_W = np.zeros((p*n, new_rank))

            new_col = 0
            for r in range(current_rank):
                if r == k:
                    continue
                scale = 1.0 - (nv[r] / c_k)
                new_U[:, new_col] = scale * current_U[:, r]
                new_V[:, new_col] = scale * current_V[:, r]
                new_W[:, new_col] = scale * current_W[:, r]
                new_col += 1

            if verify_decomposition(new_U, new_V, new_W, new_rank, atol=1e-8):
                return new_U, new_V, new_W, new_rank, True

        return current_U, current_V, current_W, current_rank, False

    # Main rank reduction loop with multiple strategies
    reduced = True
    while reduced:
        if time.time() > DEADLINE - 5:
            break

        # Try with different tolerances
        for tol in [1e-9, 1e-10, 1e-11, 1e-12]:
            current_U, current_V, current_W, current_rank, reduced = try_reduce_rank(
                current_U, current_V, current_W, current_rank, tol=tol
            )
            if reduced:
                break

    # Final high-precision polish pass
    for _ in range(50):
        if time.time() > DEADLINE - 2:
            break

        T_mat = compute_tensor_matrix(current_U, current_V, current_W, current_rank)

        try:
            U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=True)
        except:
            break

        # Use relative tolerance based on largest singular value
        if len(s) > 0 and s[0] > 0:
            rel_tol = max(1e-14, s[0] * 1e-14)
        else:
            rel_tol = 1e-14

        rank_mat = np.sum(s > rel_tol)
        if rank_mat >= current_rank:
            break

        null_vecs = Vh[rank_mat:current_rank]
        found = False
        for nv in null_vecs:
            mx = np.max(np.abs(nv))
            if mx < rel_tol:
                continue
            k = np.argmax(np.abs(nv))
            c_k = nv[k]
            if abs(c_k) < rel_tol:
                continue

            new_rank = current_rank - 1
            new_U = np.zeros((n*m, new_rank))
            new_V = np.zeros((m*p, new_rank))
            new_W = np.zeros((p*n, new_rank))

            new_col = 0
            for r in range(current_rank):
                if r == k:
                    continue
                scale = 1.0 - (nv[r] / c_k)
                new_U[:, new_col] = scale * current_U[:, r]
                new_V[:, new_col] = scale * current_V[:, r]
                new_W[:, new_col] = scale * current_W[:, r]
                new_col += 1

            if verify_decomposition(new_U, new_V, new_W, new_rank, atol=1e-10):
                current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
                found = True
                break

        if not found:
            break

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
# EVOLVE-BLOCK-END
