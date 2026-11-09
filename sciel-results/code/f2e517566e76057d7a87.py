"""Tensor decomposition for matrix multiplication 2x4 * 4x7.
Uses a structured random-search approach to find a rank-minimal decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm = n * m       # 8
    mp = m * p       # 28
    pn = p * n       # 14

    # Build the matrix multiplication tensor T as a 3D array: T[i, j, k] = 1
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

    # Schoolbook rank for fallback
    rank_schoolbook = n * m * p  # 56

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        U_mat = np.array(U, dtype=float).reshape(n, m, r)
        V_mat = np.array(V, dtype=float).reshape(m, p, r)
        W_mat = np.array(W, dtype=float).reshape(p, n, r)
        # Compute the reconstructed tensor
        recon = np.zeros((n, m, p))
        for c in range(r):
            recon += U_mat[:, :, c] * V_mat[:, :, c] * W_mat[:, :, c]
        # Check against the all-ones tensor
        target = np.ones((n, m, p))
        return np.max(np.abs(recon - target)) < tol

    def to_ints(arr, tol=1e-3):
        """Round to nearest half-integer and check consistency."""
        r = np.round(arr * 2.0) / 2.0
        return r

    def random_decomposition(rank, seed=None):
        """Generate a random decomposition attempt."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        return U, V, W

    def try_rank(rank, max_iters=500, seed=42):
        """Try to find a valid decomposition of given rank."""
        random.seed(seed)
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            # Compute residual
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            W_mat = W.reshape(p, n, rank)
            recon = np.zeros((n, m, p))
            for c in range(rank):
                recon += U_mat[:, :, c] * V_mat[:, :, c] * W_mat[:, :, c]
            target = np.ones((n, m, p))
            resid = recon - target

            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            dU[i, j, c] += resid[i, j, k] * V_mat[j, k, c] * W_mat[k, i, c]
                            dV[j, k, c] += resid[i, j, k] * U_mat[i, j, c] * W_mat[k, i, c]
                            dW[k, i, c] += resid[i, j, k] * U_mat[i, j, c] * V_mat[j, k, c]

            # Learning rate schedule
            lr = 0.1 / (1.0 + it * 0.01)
            U -= lr * dU.reshape(nm, rank)
            V -= lr * dV.reshape(mp, rank)
            W -= lr * dW.reshape(pn, rank)

            # Check convergence
            if np.max(np.abs(resid)) < 1e-4:
                # Try to snap to half-integers
                U_r = to_ints(U)
                V_r = to_ints(V)
                W_r = to_ints(W)
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), tol=1e-3):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                # Return the float version if integer snap fails but float works
                if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                    return U.tolist(), V.tolist(), W.tolist()

        return None

    # Try ranks from low to high
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # For 2x4x7, let's try from 28 upward
    for rank in range(28, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
        result = try_rank(rank, max_iters=200, seed=rank * 1000)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
