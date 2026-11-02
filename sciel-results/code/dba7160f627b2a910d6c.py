"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True


def _solve_linear(n, m, p, rank, seed=42):
    """Solve the linear system for a rank-r decomposition using fixed W."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Fix W to random half-integer values
    W = np.zeros((W_size, rank))
    for idx in range(W_size):
        for c in range(rank):
            if rng.random() < 0.7:
                W[idx, c] = rng.choice([-1, -0.5, 0.5, 1])
    
    # For each (i,j,k), the equation is:
    # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is bilinear in U and V, so it's not linear.
    # Instead, fix V and solve for U, or use a different approach.
    
    # Alternative: use the fact that we can set up a system where we fix two of the three matrices.
    # Fix U and V, solve for W. But W has 26*r unknowns and 52 equations.
    
    # Better: use a greedy approach. Set columns one at a time.
    # For column c, choose U_c, V_c, W_c such that they help satisfy the equations.
    
    # Let's try: fix U and V to simple patterns, solve for W.
    U = np.zeros((U_size, rank))
    V = np.zeros((V_size, rank))
    
    # Set U and V to identity-like patterns
    for c in range(rank):
        i = c % n
        j = (c // n) % m
        k = (c // (n*m)) % p
        U[i*m+j, c] = 1
        V[j*p+k, c] = 1
    
    # Now solve for W: for each (i,j,k):
    # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in W.
    
    # Build the linear system
    num_eqs = n * m * p
    A = np.zeros((num_eqs, W_size * rank))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    coeff = U[i*m+j, c] * V[j*p+k, c]
                    if abs(coeff) > 1e-9:
                        A[eq_idx, k*n+i * rank + c] = coeff
                eq_idx += 1
    
    # Solve A @ x = b
    try:
        x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution is close to exact
    residual = np.max(np.abs(A @ x - b))
    if residual > 1e-9:
        return None
    
    # Reshape x to W
    W_sol = x.reshape(W_size, rank)
    
    # Check if entries are close to half-integers
    W_rounded = np.round(W_sol * 2) / 2
    
    # Verify
    W_list = [[float(W_rounded[idx, c]) for c in range(rank)] for idx in range(W_size)]
    U_list = [[float(U[idx, c]) for c in range(rank)] for idx in range(U_size)]
    V_list = [[float(V[idx, c]) for c in range(rank)] for idx in range(V_size)]
    
    if _verify(n, m, p, U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    n, m, p = DIMENSIONS

    # Schoolbook is always valid.
    result = _build_schoolbook(n, m, p)
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        pass

    best_result = result
    best_rank = len(result["U"][0])
    
    # Try linear solve approach for various ranks
    for rank in range(26, best_rank):
        if time.time() > deadline - 30:
            break
        candidate = _solve_linear(n, m, p, rank, seed=rank)
        if candidate is not None:
            best_result = candidate
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END

