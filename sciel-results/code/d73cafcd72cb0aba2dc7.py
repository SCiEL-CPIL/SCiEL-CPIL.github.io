"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 145  # 5s safety margin for 150s limit

    # Build the matrix multiplication tensor T[a,b,c]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 if j==j', i==i', k==k', else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that sum_t U[:,t] * V[:,t] * W[:,t] == T"""
        # Compute the tensor from the decomposition
        T_recon = np.einsum('at,bt,ct->abc', U_mat, V_mat, W_mat)
        return np.allclose(T_recon, T, atol=tol)

    def try_random_decomposition(rank, n_trials=200, seed=42):
        """Try to find a rank-r decomposition using random initialization"""
        rng = np.random.RandomState(seed)
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            # Random initialization with small values
            U_mat = rng.randn(n * m, rank) * 0.1
            V_mat = rng.randn(m * p, rank) * 0.1
            W_mat = rng.randn(p * n, rank) * 0.1
            
            if verify(U_mat, V_mat, W_mat, tol=1e-8):
                return U_mat, V_mat, W_mat
        return None

    def try_structured_decomposition(rank):
        """Try a structured approach based on the matmul structure"""
        # For each (j,k), we have a term. Try to group them.
        # This is complex, so skip for now.
        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound: at least max(n, m, p) = 10, but likely higher
    # Let's try from 30 (max of n*m, m*p, p*n) up to 60
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        result = try_random_decomposition(rank, n_trials=50, seed=rank)
        if result is not None:
            U_mat, V_mat, W_mat = result
            return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
