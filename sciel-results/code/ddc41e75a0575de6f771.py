"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 32

    deadline = time.time() + 125

    # Schoolbook as guaranteed fallback
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
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_school

    # Precompute index arrays for vectorized tensor evaluation
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    U_idx = (i_vals * m + j_vals).flatten()
    V_idx = (j_vals * p + k_vals).flatten()
    W_idx = (k_vals * n + i_vals).flatten()
    num_eqs = n * m * p

    def verify(U, V, W):
        """Check if (U,V,W) exactly decomposes the tensor (within tolerance)."""
        vals = np.zeros(num_eqs)
        for r in range(U.shape[1]):
            vals += U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
        return np.max(np.abs(vals - 1.0)) < 1e-9

    def round_half(U, V, W):
        """Round to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, True
        return U, V, W, False

    # Try to find lower-rank decompositions
    # Use random initialization + gradient descent, then round to half-integers
    for rank in range(24, rank_school):
        if time.time() > deadline:
            break

        max_attempts = 40
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            seed = attempt * 104729 + rank * 17 + 42
            rng = np.random.RandomState(seed)

            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)

            # Gradient descent with adaptive learning rate
            lr = 0.02
            for step in range(800):
                if time.time() > deadline:
                    break

                vals = np.zeros(num_eqs)
                for r in range(rank):
                    vals += U[U_idx, r] * V[V_idx, r] * W[W_idx, r]

                err = vals - 1.0
                if np.max(np.abs(err)) < 1e-10:
                    break

                grad_U = np.zeros((n * m, rank))
                grad_V = np.zeros((m * p, rank))
                grad_W = np.zeros((p * n, rank))

                for r in range(rank):
                    cU = 2 * err * V[V_idx, r] * W[W_idx, r]
                    np.add.at(grad_U, U_idx, cU)
                    cV = 2 * err * U[U_idx, r] * W[W_idx, r]
                    np.add.at(grad_V, V_idx, cV)
                    cW = 2 * err * U[U_idx, r] * V[V_idx, r]
                    np.add.at(grad_W, W_idx, cW)

                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

                # Adaptive learning rate
                if step % 100 == 99:
                    lr *= 0.8

            # Try rounding to half-integers
            U_r, V_r, W_r, ok = round_half(U, V, W)
            if ok and rank < best_rank:
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                break

    U_final, V_final, W_final = best_result

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
