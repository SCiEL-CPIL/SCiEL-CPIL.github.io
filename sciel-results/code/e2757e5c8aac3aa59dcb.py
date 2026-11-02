import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    # Build the known rank-56 Strassen solution
    rank_56 = 56
    U56 = [[0] * rank_56 for _ in range(n * m)]
    V56 = [[0] * rank_56 for _ in range(m * p)]
    W56 = [[0] * rank_56 for _ in range(p * n)]

    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U56[li][col] = u_vecs[s][li]
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V56[v_indices[li]][col] = v_vecs[s][li]
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W56[w_indices[li]][col] = w_vecs[s][li]

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0

    def _verify(U, V, W, atol=1e-6):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        return np.allclose(T_recon, T, atol=atol)

    # Verify rank-56 solution
    if not _verify(U56, V56, W56):
        return {"U": U56, "V": V56, "W": W56}

    def _als_solve(r, U_init, V_init, W_init, max_iter=50, tol=1e-8):
        """ALS optimization for rank-r decomposition."""
        Ua = np.array(U_init, dtype=float).copy()
        Va = np.array(V_init, dtype=float).copy()
        Wa = np.array(W_init, dtype=float).copy()

        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None, None, None

            # Update U
            KR_VW = np.einsum('bs,cs->bcs', Va, Wa).reshape(m*p*p*n, r)
            for a in range(n*m):
                T_a = T[a].reshape(-1)
                Ua[a, :], _, _, _ = np.linalg.lstsq(KR_VW, T_a, rcond=None)

            # Update V
            KR_UW = np.einsum('as,cs->acs', Ua, Wa).reshape(n*m*p*n, r)
            for b in range(m*p):
                T_b = T[:, b, :].reshape(-1)
                Va[b, :], _, _, _ = np.linalg.lstsq(KR_UW, T_b, rcond=None)

            # Update W
            KR_UV = np.einsum('as,bs->abs', Ua, Va).reshape(n*m*m*p, r)
            for c in range(p*n):
                T_c = T[:, :, c].reshape(-1)
                Wa[c, :], _, _, _ = np.linalg.lstsq(KR_UV, T_c, rcond=None)

            # Check convergence
            T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break

        # Final check
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        if np.allclose(T_recon, T, atol=1e-5):
            return Ua.tolist(), Va.tolist(), Wa.tolist()
        return None, None, None

    # Try to find lower rank solutions
    best_U, best_V, best_W = U56, V56, W56
    best_rank = rank_56

    for target_rank in range(55, 40, -1):
        if time.time() > deadline - 30:
            break

        # Initialize from rank-56 by projecting
        U56_a = np.array(U56, dtype=float)
        V56_a = np.array(V56, dtype=float)
        W56_a = np.array(W56, dtype=float)

        # Try multiple initializations
        for init_seed in range(3):
            if time.time() > deadline - 20:
                break

            np.random.seed(init_seed + 1000)

            if target_rank == 55:
                # Remove one column and re-optimize
                U_init = U56_a[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.01
                V_init = V56_a[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.01
                W_init = W56_a[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.01
            else:
                # Random projection
                proj_U = np.random.randn(n*m, target_rank) * 0.1
                proj_V = np.random.randn(m*p, target_rank) * 0.1
                proj_W = np.random.randn(p*n, target_rank) * 0.1

                U_init = proj_U
                V_init = proj_V
                W_init = proj_W

            U_res, V_res, W_res = _als_solve(target_rank, U_init, V_init, W_init, max_iter=30)

            if U_res is not None:
                best_U, best_V, best_W = U_res, V_res, W_res
                best_rank = target_rank
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
