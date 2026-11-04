"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 105  # safety margin before 120s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k', k*n+i'] = 1 iff j==j' and i==i' and k==k'
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if the decomposition exactly reproduces the tensor."""
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Efficient computation using einsum
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=1e-9)

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

    # Verify schoolbook is correct
    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    # Try to find a lower-rank decomposition using numerical optimization
    # Strategy: use SVD on matricizations to get initial guess, then refine
    
    best_rank = rank_sch
    best_U = U_sch
    best_V = V_sch
    best_W = W_sch

    # Try various ranks using alternating least squares / projection
    # We use random initialization because SVD of matricizations is limited by mode dimensions
    # which may be smaller than the target rank.
    
    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    # Try ranks from 54 to 71 (skip if too slow)
    # Note: We only try a few ranks to stay within time limits
    for target_rank in [60, 65, 70]:
        if time.time() > deadline:
            break

        # Random initialization
        np.random.seed(42)
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)

        # Refine using alternating least squares
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            MW = np.einsum('br,cr->bcr', V_init, W_init)  # (m*p, p*n, r)
            MW_flat = MW.transpose(2, 0, 1).reshape(target_rank, m * p * p * n)
            try:
                G = MW_flat @ MW_flat.T
                G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                U_init = T_A @ MW_flat.T @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update V
            UW = np.einsum('ar,cr->acr', U_init, W_init)  # (n*m, p*n, r)
            UW_flat = UW.transpose(2, 0, 1).reshape(target_rank, n * m * p * n)
            try:
                G = UW_flat @ UW_flat.T
                G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                V_init = T_B @ UW_flat.T @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update W
            UV = np.einsum('ar,br->abr', U_init, V_init)  # (n*m, m*p, r)
            UV_flat = UV.transpose(2, 0, 1).reshape(target_rank, n * m * m * p)
            try:
                G = UV_flat @ UV_flat.T
                G_inv = np.linalg.inv(G + 1e-10 * np.eye(target_rank))
                W_init = T_C @ UV_flat.T @ G_inv
            except np.linalg.LinAlgError:
                break

        # Check if we found an exact solution
        if verify(U_init, V_init, W_init):
            # Round to integers/half-integers
            U_rounded = np.round(U_init * 2) / 2
            V_rounded = np.round(V_init * 2) / 2
            W_rounded = np.round(W_init * 2) / 2
            if verify(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break  # Found a valid lower-rank decomposition

    # If no better found, use schoolbook
    if best_rank == rank_sch:
        U_out = U_sch
        V_out = V_sch
        W_out = W_sch
    else:
        U_out = best_U
        V_out = best_V
        W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
