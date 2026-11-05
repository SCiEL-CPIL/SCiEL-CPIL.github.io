import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
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

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()

    # Verify rank 50
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

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        """
        Compute Khatri-Rao product of A (M x R) and B (N x R).
        Result is (M*N) x R.
        """
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=300, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition with improved convergence.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for it in range(max_iters):
            # Solve for U
            KR_VW = khatri_rao(V, W)
            try:
                U_new = T1 @ np.linalg.pinv(KR_VW)
            except:
                break

            # Solve for V
            KR_WU = khatri_rao(W, U)
            try:
                V_new = T2 @ np.linalg.pinv(KR_WU)
            except:
                break

            # Solve for W
            KR_UV = khatri_rao(U, V)
            try:
                W_new = T3 @ np.linalg.pinv(KR_UV)
            except:
                break

            # Check convergence via residual
            recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            residual = np.linalg.norm(recon - T)

            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new

            if change < tol or residual < 1e-10:
                break

        return U, V, W

    def polish_decomposition(U, V, W, max_iters=500, tol=1e-14):
        """
        High-precision polish: alternate between ALS steps with tighter tolerance.
        """
        R = U.shape[1]
        for _ in range(max_iters):
            # Standard ALS step
            KR_VW = khatri_rao(V, W)
            U = T1 @ np.linalg.pinv(KR_VW)

            KR_WU = khatri_rao(W, U)
            V = T2 @ np.linalg.pinv(KR_WU)

            KR_UV = khatri_rao(U, V)
            W = T3 @ np.linalg.pinv(KR_UV)

            # Check if we're close
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break

        return U, V, W

    def final_refinement(U, V, W):
        """
        Final high-precision refinement using iterative least squares on residuals.
        """
        R = U.shape[1]
        for _ in range(100):
            # Compute residual
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual_tensor = T - recon

            # Try to correct by adjusting W (smallest dimension)
            KR_UV = khatri_rao(U, V)
            W = T3 @ np.linalg.pinv(KR_UV)

            KR_VW = khatri_rao(V, W)
            U = T1 @ np.linalg.pinv(KR_VW)

            KR_WU = khatri_rao(W, U)
            V = T2 @ np.linalg.pinv(KR_WU)

            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-14:
                break

        return U, V, W

    # Try to reduce rank
    # Start from 49 down to 28
    # Use time budget to decide how many restarts

    rng = np.random.default_rng(42)

    for target_rank in range(49, 27, -1):
        if time.time() > deadline - 10:
            break

        # Determine number of restarts based on time
        # More restarts for higher ranks, fewer for lower
        num_restarts = 12 if target_rank > 40 else 8
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break

            # Initialize with multiple strategies
            if restart == 0 and target_rank == 49:
                # Drop 1 column from rank-50
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.01, U_init.shape)
                V_init += rng.normal(0, 0.01, V_init.shape)
                W_init += rng.normal(0, 0.01, W_init.shape)
            elif restart == 1:
                # Drop columns from rank-50 with different pattern
                drop_idx = rng.choice(50, size=50 - target_rank, replace=False)
                keep_idx = np.setdiff1d(np.arange(50), drop_idx)
                U_init = U50[:, keep_idx].copy()
                V_init = V50[:, keep_idx].copy()
                W_init = W50[:, keep_idx].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            elif restart < 4:
                # Random init
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            elif restart < 7:
                # Perturb rank-50 more aggressively
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.1, U_init.shape)
                V_init += rng.normal(0, 0.1, V_init.shape)
                W_init += rng.normal(0, 0.1, W_init.shape)
            else:
                # Structured init: use Strassen-like patterns
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1
                    V_init[j * p + k, r] = 1
                    W_init[k * n + i, r] = 1
                    if r < target_rank - 4:
                        U_init[i * m + j, r] += rng.normal(0, 0.1)
                        V_init[j * p + k, r] += rng.normal(0, 0.1)
                        W_init[k * n + i, r] += rng.normal(0, 0.1)

            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=200)

            # If close, try to polish
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-5:
                U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol)
                # Final refinement
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)
                if residual < 1e-8:
                    U_sol, V_sol, W_sol = final_refinement(U_sol, V_sol, W_sol)

            # Verify
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break

        if found:
            # We found a valid decomposition for this rank.
            # Try to go lower?
            # The loop continues, so it will try target_rank - 1 next.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
