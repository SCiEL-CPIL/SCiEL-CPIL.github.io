import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s limit

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

    def get_schoolbook():
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
        return U, V, W

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
        U, V, W = get_schoolbook()
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

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=200, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
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
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    def try_round_and_verify(U, V, W):
        """Try rounding to nearest 0.5 and verify"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomp(U_r, V_r, W_r, atol=1e-6):
            return U_r, V_r, W_r, True
        return U, V, W, False

    def try_integer_round_and_verify(U, V, W):
        """Try rounding to nearest integer and verify"""
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify_decomp(U_r, V_r, W_r, atol=1e-6):
            return U_r, V_r, W_r, True
        return U, V, W, False

    def greedy_rank_reduction(U, V, W, current_rank):
        """
        Try to greedily remove columns that are linearly dependent or have small norm.
        Returns a reduced-rank decomposition if found.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = current_rank
        
        # Try removing one column at a time
        for remove_idx in range(R):
            if time.time() > deadline - 10:
                break
            
            # Remove column remove_idx
            mask = [i for i in range(R) if i != remove_idx]
            U_reduced = U[:, mask]
            V_reduced = V[:, mask]
            W_reduced = W[:, mask]
            
            if verify_decomp(U_reduced, V_reduced, W_reduced):
                return U_reduced, V_reduced, W_reduced, R - 1
        
        return None, None, None, current_rank

    rng = np.random.default_rng(42)

    # Strategy 1: Try greedy removal from rank 50
    U_curr, V_curr, W_curr = U50.copy(), V50.copy(), W50.copy()
    curr_rank = 50
    
    # Try multiple rounds of greedy removal
    for round_num in range(5):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new, new_rank = greedy_rank_reduction(U_curr, V_curr, W_curr, curr_rank)
        if U_new is not None:
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            curr_rank = new_rank
            best_rank = min(best_rank, curr_rank)
            best_U, best_V, best_W = U_curr, V_curr, W_curr
        else:
            break

    # Strategy 2: ALS with better initializations
    for target_rank in range(min(49, best_rank - 1), 30, -1):
        if time.time() > deadline - 15:
            break

        remaining = deadline - time.time()
        max_restarts = min(8, max(2, int(remaining / 8)))

        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            if restart == 0:
                # Drop columns from current best
                idx = rng.choice(curr_rank, target_rank, replace=False)
                U_init = U_curr[:, idx].copy()
                V_init = V_curr[:, idx].copy()
                W_init = W_curr[:, idx].copy()
            elif restart == 1:
                # Random init with small values
                U_init = rng.normal(0, 0.5, (n*m, target_rank))
                V_init = rng.normal(0, 0.5, (m*p, target_rank))
                W_init = rng.normal(0, 0.5, (p*n, target_rank))
            elif restart == 2:
                # Structured init based on Strassen pattern
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    U_init[:, r] = rng.choice([-1, 0, 1], size=n*m)
                    V_init[:, r] = rng.choice([-1, 0, 1], size=m*p)
                    W_init[:, r] = rng.choice([-1, 0, 1], size=p*n)
            elif restart == 3:
                # Perturbed subset of rank-50
                idx = rng.choice(50, min(50, target_rank), replace=False)
                U_init = U50[:, idx].copy()
                V_init = V50[:, idx].copy()
                W_init = W50[:, idx].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            else:
                # Random sparse init
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    nnz_u = rng.integers(2, 5)
                    nnz_v = rng.integers(2, 6)
                    nnz_w = rng.integers(2, 5)
                    u_idx = rng.choice(n*m, nnz_u, replace=False)
                    v_idx = rng.choice(m*p, nnz_v, replace=False)
                    w_idx = rng.choice(p*n, nnz_w, replace=False)
                    U_init[u_idx, r] = rng.choice([-1, 1], nnz_u)
                    V_init[v_idx, r] = rng.choice([-1, 1], nnz_v)
                    W_init[w_idx, r] = rng.choice([-1, 1], nnz_w)

            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=100)

            # Verify directly
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                curr_rank = target_rank
                found = True
                break
            else:
                # Try rounding to 0.5
                U_r, V_r, W_r, success = try_round_and_verify(U_sol, V_sol, W_sol)
                if success:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    curr_rank = target_rank
                    found = True
                    break
                else:
                    # Try integer rounding
                    U_r, V_r, W_r, success = try_integer_round_and_verify(U_sol, V_sol, W_sol)
                    if success:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        curr_rank = target_rank
                        found = True
                        break

        if found:
            continue

    # Final: try to reduce further from current best using greedy
    if time.time() < deadline - 10:
        U_curr, V_curr, W_curr = best_U.copy(), best_V.copy(), best_W.copy()
        curr_rank = best_rank
        for _ in range(3):
            U_new, V_new, W_new, new_rank = greedy_rank_reduction(U_curr, V_curr, W_curr, curr_rank)
            if U_new is not None:
                U_curr, V_curr, W_curr = U_new, V_new, W_new
                curr_rank = new_rank
                best_rank = min(best_rank, curr_rank)
                best_U, best_V, best_W = U_curr, V_curr, W_curr
            else:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
