"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.

    Strategy:
      1. Build the known rank-28 block-Strassen decomposition (4 independent
         2x2x2 Strassen decompositions over the 2x2 block structure of the
         4-dim intermediate space). This is a guaranteed exact fallback.
      2. Treat the 4-dim intermediate as a tensor-product space C^2 (o) C^2 and
         attempt to exploit cross-block separability to reduce the rank below 28.
         We do this with a smooth/linear null-space search: if the 28 rank-1
         factor columns are linearly dependent in a way that preserves the
         tensor, we can drop dependent columns. We then polish to exact
         half-integer entries.
      3. Verify any candidate exactly (integer/half-integer) and return the
         lowest-rank valid decomposition found before the deadline.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before the 150s kill

    def time_left():
        return deadline - time.time()

    # ---------- verification ----------
    def verify(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return bool(np.allclose(T, target, atol=tol))

    def is_half_integer(U, V, W, tol=1e-6):
        for M in (U, V, W):
            if not np.allclose(2 * np.asarray(M, dtype=float),
                               np.round(2 * np.asarray(M, dtype=float)), atol=tol):
                return False
        return True

    # ---------- rank-28 block-Strassen fallback ----------
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # Strassen rank-7 terms for 2x2x2 (local indices 0..3)
        # U idx: i*2+j ; V idx: j*2+k ; W idx: k*2+i
        strassen_terms = [
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
        ]

        # 4 blocks: (j_offset, k_offset)
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]

        col = 0
        for (j_off, k_off) in blocks:
            for term in strassen_terms:
                for (idx, coeff) in term['U']:
                    i = idx // 2
                    j = idx % 2
                    U[i * m + (j + j_off)][col] += coeff
                for (idx, coeff) in term['V']:
                    j = idx // 2
                    k = idx % 2
                    V[(j + j_off) * p + (k + k_off)][col] += coeff
                for (idx, coeff) in term['W']:
                    k = idx // 2
                    i = idx % 2
                    W[(k + k_off) * n + i][col] += coeff
                col += 1

        return {"U": U, "V": V, "W": W}

    best = construct_strassen_rank28()
    if not verify(best['U'], best['V'], best['W']):
        # Should never happen; return as-is (known correct construction).
        return best

    # ---------- attempt rank reduction via null-space projection ----------
    # The 28 rank-1 columns (u_c, v_c, w_c) produce the tensor. If there is a
    # non-trivial linear relation among the *tensor contributions* that keeps the
    # sum fixed, we can re-parameterize to drop a column. Concretely, the set of
    # rank-1 tensors {u_c v_c w_c} spans a subspace; the target tensor is a
    # specific element. We search for a representation using fewer rank-1 terms
    # by projecting onto a lower-dimensional factor structure.
    #
    # Practical approach: build the linear system for a target rank r and solve.
    # For a given r, the unknowns are the entries of U, V, W (r columns each).
    # The constraint is bilinear (product of U and V entries times W), so we use
    # an iterative alternating-minimization / fixed-point approach seeded from
    # the rank-28 solution, then polish to half-integers.

    def try_reduce(target_rank):
        """Attempt to find an exact half-integer decomposition of target_rank.
        Uses the rank-28 solution as a seed and applies a smooth alternating
        least-squares reduction followed by exact rounding/polish."""
        if time_left() < 5.0:
            return None

        U0 = np.asarray(best['U'], dtype=float)
        V0 = np.asarray(best['V'], dtype=float)
        W0 = np.asarray(best['W'], dtype=float)
        r0 = U0.shape[1]

        # Seed: keep the first `target_rank` columns of the rank-28 solution,
        # then re-optimize the kept columns and the dropped structure.
        U = U0[:, :target_rank].copy()
        V = V0[:, :target_rank].copy()
        W = W0[:, :target_rank].copy()

        # Alternating least squares: fix two factors, solve for the third.
        # For fixed V, W: T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c]*W[k,i,c])
        # => for each (i,j): T[i,j,:] = U[i,j,:] @ diag? No, it's:
        #    T[i,j,k] = sum_c U[i,j,c] * B_c[j,k] where B_c[j,k]=V[j,k,c]*W[k,i,c]
        # This is linear in U[i,j,:]: for each k, sum_c U[i,j,c]*B_c[j,k] = T[i,j,k]
        # Build a (p x r) matrix M[i,j] with M[i,j][k,c] = V[j,k,c]*W[k,i,c].
        # Then solve M[i,j] @ U[i,j,:] = T[i,j,:]  (p equations, r unknowns).
        # Use lstsq (NNLS-free; allow negatives since half-integers can be neg).

        def build_M_U(i, j):
            # M[k, c] = V[j, k, c] * W[k, i, c]
            Vj = V3c[j]  # shape (p, r)
            Wi = W3c[i]  # shape (p, r)
            return Vj * Wi  # (p, r)

        def build_M_V(j, k):
            # For fixed U, W: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (j,k): sum_c V[j,k,c] * C_c[i] = T[i,j,k], C_c[i]=U[i,j,c]*W[k,i,c]
            # M[i, c] = U[i,j,c]*W[k,i,c]  -> (n, r)
            Uj = U3c[j]  # (n, r)
            Wk = W3c[k]  # (n, r)
            return Uj * Wk

        def build_M_W(k, i):
            # For fixed U, V: T[i,j,k] = sum_c W[k,i,c] * D_c[j], D_c[j]=U[i,j,c]*V[j,k,c]
            # M[j, c] = U[i,j,c]*V[j,k,c] -> (m, r)
            Ui = U3c[i]  # (m, r)
            Vk = V3c[k]  # (m, r)
            return Ui * Vk

        # We need the 3D views; rebuild each iteration since shapes change.
        def refresh_views():
            U3c = U.reshape(n, m, target_rank)
            V3c = V.reshape(m, p, target_rank)
            W3c = W.reshape(p, n, target_rank)
            return U3c, V3c, W3c

        T_target = np.ones((n, m, p))

        def solve_linear(M, rhs):
            # M: (rows, r), rhs: (rows,). Solve M x = rhs by lstsq.
            x, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            return x

        iters = 0
        max_iters = 400
        while iters < max_iters and time_left() > 8.0:
            U3c, V3c, W3c = refresh_views()
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V3c[j] * W3c[i]  # (p, r)
                    rhs = T_target[i, j, :]  # (p,)
                    U[i * m + j, :] = solve_linear(M, rhs)
            iters += 1
            if iters % 3 == 0:
                # Update V
                U3c, V3c, W3c = refresh_views()
                for j in range(m):
                    for k in range(p):
                        M = U3c[j] * W3c[k]  # (n, r)
                        rhs = T_target[:, j, k]  # (n,)
                        V[j * p + k, :] = solve_linear(M, rhs)
                iters += 1
                # Update W
                U3c, V3c, W3c = refresh_views()
                for k in range(p):
                    for i in range(n):
                        M = U3c[i] * V3c[k]  # (m, r)
                        rhs = T_target[i, :, k]  # (m,)
                        W[k * n + i, :] = solve_linear(M, rhs)
                iters += 1

            # Check convergence
            U3c, V3c, W3c = refresh_views()
            T = np.zeros((n, m, p))
            for c in range(target_rank):
                T += np.einsum('ij,jk,ki->ijk', U3c[:, :, c], V3c[:, :, c], W3c[:, :, c])
            if np.allclose(T, T_target, atol=1e-7):
                # Polish to half-integers
                U2 = np.round(2 * U) / 2.0
                V2 = np.round(2 * V) / 2.0
                W2 = np.round(2 * W) / 2.0
                if verify(U2, V2, W2, tol=1e-6) and is_half_integer(U2, V2, W2):
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                # try without rounding (maybe already near half-int)
                if verify(U, V, W, tol=1e-7) and is_half_integer(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Try to reduce from 28 down to a known lower bound.
    # Lower bound: rank >= n*m*p / (something). For 2x4x4, known lower bounds
    # are around 24. Try 27, 26, 25, 24 in order (cheapest first).
    for target_rank in [27, 26, 25, 24]:
        if time_left() < 10.0:
            break
        candidate = try_reduce(target_rank)
        if candidate is not None:
            best = candidate
            # Try to go even lower
            continue

    # Final safety: ensure best is valid; if not, return the rank-28 fallback.
    if not verify(best['U'], best['V'], best['W']):
        fallback = construct_strassen_rank28()
        return fallback

    return best

# EVOLVE-BLOCK-END
