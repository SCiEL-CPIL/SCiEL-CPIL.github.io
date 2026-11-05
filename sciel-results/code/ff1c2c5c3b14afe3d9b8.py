"""Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication.

Uses numerical ALS optimization to find low-rank decompositions.
Index conventions:
  U has shape (n*m, rank) = (12, rank), indexed by [i*m + j]
  V has shape (m*p, rank) = (24, rank), indexed by [j*p + k]
  W has shape (p*n, rank) = (18, rank), indexed by [k*n + i]

Tensor: T[i*m+j, j'*p+k', k''*n+i''] = 1 iff j==j' and i==i' and k==k'
"""
import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 115  # safety margin before 120s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=tol)

    def round_to_half(U, V, W):
        """Round to integers/half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-8):
            return U_r, V_r, W_r
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_step(U, V, W, rank, T_A, T_B, T_C, n, m, p):
        """One ALS iteration. Returns updated U, V, W."""
        # Update U: solve for U given V, W
        MW = np.einsum('br,cr->bcr', V, W)
        MW_flat = MW.transpose(2, 0, 1).reshape(rank, m * p * p * n)
        G = MW_flat @ MW_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            U_new = T_A @ MW_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U, V, W

        # Update V: solve for V given U, W
        UW = np.einsum('ar,cr->acr', U_new, W)
        UW_flat = UW.transpose(2, 0, 1).reshape(rank, n * m * p * n)
        G = UW_flat @ UW_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            V_new = T_B @ UW_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U_new, V, W

        # Update W: solve for W given U, V
        UV = np.einsum('ar,br->abr', U_new, V_new)
        UV_flat = UV.transpose(2, 0, 1).reshape(rank, n * m * m * p)
        G = UV_flat @ UV_flat.T
        G += 1e-12 * np.eye(rank)
        try:
            W_new = T_C @ UV_flat.T @ np.linalg.solve(G, np.eye(rank))
        except np.linalg.LinAlgError:
            return U_new, V_new, W

        return U_new, V_new, W_new

    def compute_residual(U, V, W):
        """Compute the residual norm."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_computed - T)

    def svd_init(rank, seed):
        """Initialize U, V, W using SVD of matricizations."""
        np.random.seed(seed)
        # Random subspace projection to get a rank-r approximation of each matricization
        # This is a standard technique for tensor SVD initialization
        
        # For U (mode 1)
        proj_VW = np.random.randn(m * p * p * n, rank)
        T_A_proj = T_A @ proj_VW
        U_init = np.linalg.svd(T_A_proj, full_matrices=False)[0][:, :rank]
        
        # For V (mode 2)
        proj_UW = np.random.randn(n * m * p * n, rank)
        T_B_proj = T_B @ proj_UW
        V_init = np.linalg.svd(T_B_proj, full_matrices=False)[0][:, :rank]
        
        # For W (mode 3)
        proj_UV = np.random.randn(n * m * m * p, rank)
        T_C_proj = T_C @ proj_UV
        W_init = np.linalg.svd(T_C_proj, full_matrices=False)[0][:, :rank]
        
        return U_init, V_init, W_init

    def polish(U, V, W, rank):
        """Polish the solution using scipy.optimize.least_squares."""
        def residual_func(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            T_computed = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            return (T_computed - T).flatten()
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use a limited number of iterations to keep it fast
        try:
            result = scipy.optimize.least_squares(residual_func, x0, max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.success or result.cost < 1e-10:
                U_p = result.x[:n*m*rank].reshape(n*m, rank)
                V_p = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_p = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                return U_p, V_p, W_p
        except Exception:
            pass
        return U, V, W

    # Try to find lower-rank decompositions
    # Start with promising ranks based on known bounds
    # We will try ranks from 60 down to 40 in steps of 2 or 1
    # Given the 120s limit, we need to be efficient.
    target_ranks = list(range(60, 39, -1))

    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        # Multiple random initializations
        for seed in range(20):
            if time.time() > deadline:
                break

            # Use SVD initialization
            U_cur, V_cur, W_cur = svd_init(target_rank, seed)
            
            # Run ALS
            prev_res = np.inf
            for iteration in range(50):
                if time.time() > deadline:
                    break
                U_cur, V_cur, W_cur = als_step(
                    U_cur, V_cur, W_cur, target_rank, T_A, T_B, T_C, n, m, p
                )
                res = compute_residual(U_cur, V_cur, W_cur)
                if prev_res - res < 1e-14:
                    break
                prev_res = res
                if res < 1e-10:
                    break

            # Polish if residual is small but not zero
            if res < 1e-6:
                U_cur, V_cur, W_cur = polish(U_cur, V_cur, W_cur, target_rank)
                res = compute_residual(U_cur, V_cur, W_cur)

            # Final verification and rounding
            if res < 1e-8:
                rounded = round_to_half(U_cur, V_cur, W_cur)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if verify(U_r, V_r, W_r, tol=1e-8):
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        break  # Found valid decomposition for this rank

        if best_rank < rank_sch:
            # Found improvement, try to go lower
            continue
        else:
            # Didn't find anything at this rank, move to next
            continue

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
