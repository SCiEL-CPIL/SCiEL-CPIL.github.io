"""Matrix multiplication 3x4x6 - tensor decomposition with alternating projections.

Uses a structured approach:
1. Start with schoolbook rank-72 decomposition
2. Try to merge terms using alternating least squares with half-integer snapping
3. Validate exactness at each step
4. Return best found decomposition
"""
import numpy as np
import random
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T exactly (within tol)."""
        # Efficiently compute the tensor from factors
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def snap_to_half(X):
        """Snap values to nearest half-integer."""
        return np.round(2.0 * X) / 2.0

    def snap_to_int(X):
        """Snap values to nearest integer."""
        return np.round(X)

    def try_rank(rank, max_iters=200, n_restarts=5, seed=42):
        """Try to find a rank-`rank` decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)

        for restart in range(n_restarts):
            # Initialize with random factors
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5

            # Or initialize from schoolbook if rank >= schoolbook_rank
            if rank >= schoolbook_rank:
                U = np.zeros((n * m, rank), dtype=float)
                V = np.zeros((m * p, rank), dtype=float)
                W = np.zeros((p * n, rank), dtype=float)
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                # Zero out extra columns
                if rank > schoolbook_rank:
                    U[:, schoolbook_rank:] = 0
                    V[:, schoolbook_rank:] = 0
                    W[:, schoolbook_rank:] = 0
                if check_decomposition(U, V, W):
                    return U, V, W
                # Perturb to escape
                U += rng.randn(n * m, rank) * 0.1
                V += rng.randn(m * p, rank) * 0.1
                W += rng.randn(p * n, rank) * 0.1

            prev_err = float('inf')
            for it in range(max_iters):
                # Solve for U given V, W
                # For each a: U[a,:] should satisfy sum_b T[a,b,:] V[b,:] W[:,c] pattern
                # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W: for each a, we want U[a,:] such that
                # sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c] for all b,c
                # This is: U[a,:] * (V ⊙ W)^T = T[a,:,:].reshape(-1)
                # where (V ⊙ W) has entries (V[b,r]*W[c,r])
                # So: U[a,:] * M = T[a,:,:].reshape(-1) where M[bc, r] = V[b,r]*W[c,r]

                # Build M of shape (m*p * p*n, rank)
                M = np.zeros((m * p * p * n, rank))
                for b in range(m * p):
                    for c in range(p * n):
                        M[b * p * n + c, :] = V[b, :] * W[c, :]

                # Solve U: U[a, :] * M = T[a, :, :].reshape(-1)
                # U[a, :] = T[a,:,:].reshape(-1) @ pinv(M)
                M_pinv = np.linalg.pinv(M)
                U_new = np.zeros((n * m, rank))
                for a in range(n * m):
                    target = T[a, :, :].reshape(-1)
                    U_new[a, :] = target @ M_pinv
                U = U_new

                # Snap to half-integers
                U = snap_to_half(U)

                # Solve for V given U, W
                # For each b: V[b,:] * (U ⊙ W)^T = T[:,b,:].reshape(-1)
                M2 = np.zeros((n * m * p * n, rank))
                for a in range(n * m):
                    for c in range(p * n):
                        M2[a * p * n + c, :] = U[a, :] * W[c, :]
                M2_pinv = np.linalg.pinv(M2)
                V_new = np.zeros((m * p, rank))
                for b in range(m * p):
                    target = T[:, b, :].reshape(-1)
                    V_new[b, :] = target @ M2_pinv
                V = snap_to_half(V_new)

                # Solve for W given U, V
                # For each c: W[c,:] * (U ⊙ V)^T = T[:,:,c].reshape(-1)
                M3 = np.zeros((n * m * m * p, rank))
                for a in range(n * m):
                    for b in range(m * p):
                        M3[a * m * p + b, :] = U[a, :] * V[b, :]
                M3_pinv = np.linalg.pinv(M3)
                W_new = np.zeros((p * n, rank))
                for c in range(p * n):
                    target = T[:, :, c].reshape(-1)
                    W_new[c, :] = target @ M3_pinv
                W = snap_to_half(W_new)

                # Check convergence and exactness
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-9:
                    return U, V, W

                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err

        return None, None, None

    # Try ranks from a few below schoolbook down to some lower bound
    # Lower bound: max(n*m, m*p, p*n, n*p) = max(12, 24, 18, 18) = 24
    # But actual lower bound for 3x4x6 is likely higher.
    # Try a range of ranks
    best_result = None

    # First, verify schoolbook works
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    if check_decomposition(U, V, W):
        best_result = (rank, U, V, W)

    # Try to find lower rank
    # Try ranks: 71, 70, 69, ... down to maybe 50
    # But this is expensive, so try a few
    for rank in range(schoolbook_rank - 1, max(24, schoolbook_rank - 25), -1):
        if best_result is not None and rank >= best_result[0]:
            continue
        result = try_rank(rank, max_iters=50, n_restarts=3, seed=rank)
        if result[0] is not None:
            best_result = (rank, result[0], result[1], result[2])
            break

    if best_result is not None:
        rank, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
