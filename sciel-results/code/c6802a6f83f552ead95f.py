import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M2
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            # M3
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M4
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # M5
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            # M6
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M7
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Matricizations for rank lower bound
    T1f = T.reshape(n * m, -1)
    T2f = T.reshape(m * p, -1)
    T3f = T.reshape(p * n, -1)
    r1 = np.linalg.matrix_rank(T1f)
    r2 = np.linalg.matrix_rank(T2f)
    r3 = np.linalg.matrix_rank(T3f)
    lower_bound = max(r1, r2, r3)

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(T1m, T2m, T3m, U_init, V_init, W_init, max_iters=300, tol=1e-14):
        R = U_init.shape[1]
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            try:
                KR_VW = khatri_rao(V, W)
                U_new = T1m @ np.linalg.pinv(KR_VW)
                KR_WU = khatri_rao(W, U_new)
                V_new = T2m @ np.linalg.pinv(KR_WU)
                KR_UV = khatri_rao(U_new, V_new)
                W_new = T3m @ np.linalg.pinv(KR_UV)
            except:
                break
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        return U, V, W

    def try_rank(target_rank, n_restarts=8):
        if time.time() > deadline - 6:
            return None, None, None, False
        if target_rank < lower_bound:
            return None, None, None, False

        rng = np.random.default_rng(42 + target_rank)
        best_residual = np.inf
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            if time.time() > deadline - 4:
                break

            # Structured initializations
            if restart == 0:
                # Truncate rank-50
                U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.01, (n*m, target_rank))
                V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.01, (m*p, target_rank))
                W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.01, (p*n, target_rank))
            elif restart == 1:
                # Random
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif restart == 2:
                # Sparse
                U_init = rng.choice([-1, 0, 1], (n*m, target_rank)).astype(float)
                V_init = rng.choice([-1, 0, 1], (m*p, target_rank)).astype(float)
                W_init = rng.choice([-1, 0, 1], (p*n, target_rank)).astype(float)
            elif restart == 3:
                # Mix
                U_init = 0.3 * U50[:, :target_rank] + 0.7 * rng.normal(0, 1, (n*m, target_rank))
                V_init = 0.3 * V50[:, :target_rank] + 0.7 * rng.normal(0, 1, (m*p, target_rank))
                W_init = 0.3 * W50[:, :target_rank] + 0.7 * rng.normal(0, 1, (p*n, target_rank))
            elif restart == 4:
                # Scale variation
                U_init = rng.normal(0, 0.5, (n*m, target_rank))
                V_init = rng.normal(0, 2.0, (m*p, target_rank))
                W_init = rng.normal(0, 1.0, (p*n, target_rank))
            elif restart == 5:
                # Border-rank style: perturb tail terms
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                # Add small perturbation to encourage different limit
                U_init += rng.normal(0, 0.05, (n*m, target_rank))
                V_init += rng.normal(0, 0.05, (m*p, target_rank))
                W_init += rng.normal(0, 0.05, (p*n, target_rank))
            elif restart == 6:
                # Diagonal-dominant
                U_init = np.diag(rng.normal(0, 1, min(n*m, target_rank))).reshape(n*m, -1)[:, :target_rank] if n*m >= target_rank else rng.normal(0,1,(n*m,target_rank))
                V_init = np.diag(rng.normal(0, 1, min(m*p, target_rank))).reshape(m*p, -1)[:, :target_rank] if m*p >= target_rank else rng.normal(0,1,(m*p,target_rank))
                W_init = np.diag(rng.normal(0, 1, min(p*n, target_rank))).reshape(p*n, -1)[:, :target_rank] if p*n >= target_rank else rng.normal(0,1,(p*n,target_rank))
            else:
                # Different seed
                rng2 = np.random.default_rng(1000 + restart * 7)
                U_init = rng2.normal(0, 1, (n*m, target_rank))
                V_init = rng2.normal(0, 1, (m*p, target_rank))
                W_init = rng2.normal(0, 1, (p*n, target_rank))

            U_sol, V_sol, W_sol = als_solve(T1f, T2f, T3f, U_init, V_init, W_init, max_iters=200)

            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)

            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()

            if residual < 1e-9:
                # Polish with high precision
                for _ in range(100):
                    if time.time() > deadline - 2:
                        break
                    try:
                        KR_VW = khatri_rao(V_sol, W_sol)
                        U_sol = T1f @ np.linalg.pinv(KR_VW)
                        KR_WU = khatri_rao(W_sol, U_sol)
                        V_sol = T2f @ np.linalg.pinv(KR_WU)
                        KR_UV = khatri_rao(U_sol, V_sol)
                        W_sol = T3f @ np.linalg.pinv(KR_UV)
                    except:
                        break

                if verify_decomp(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol, True

        # Check best
        if best_U is not None and best_residual < 1e-8:
            if verify_decomp(best_U, best_V, best_W):
                return best_U, best_V, best_W, True

        return None, None, None, False

    # Try ranks from 49 down to lower_bound
    for target_rank in range(49, lower_bound - 1, -1):
        if time.time() > deadline - 10:
            break

        time_left = deadline - 8 - time.time()
        if time_left < 2:
            break

        # Allocate more restarts for higher ranks (more likely to succeed)
        if target_rank >= 47:
            n_restarts = min(10, max(5, int(time_left / 3)))
        else:
            n_restarts = min(8, max(3, int(time_left / 4)))

        U_sol, V_sol, W_sol, success = try_rank(target_rank, n_restarts)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            break  # Found the best we can, stop

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
