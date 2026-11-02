"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_float(U, V, W, n, m, p, tol=1e-6):
        """Fast float verification."""
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        # Compute the tensor decomposition
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n

                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = np.dot(U_np[a], V_np[b] * W_np[c])
                        if abs(s - 1.0) > tol:
                            return False
                    else:
                        s = np.dot(U_np[a], V_np[b] * W_np[c])
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]

        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n

                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                        if s != 0:
                            return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Key insight: The matrix multiplication tensor has a structure that can be
    # exploited using a Kronecker product approach. For 2x3 * 3x7, we can try
    # to find a rank-r decomposition by using the following strategy:
    #
    # The tensor T(i,j,k) = 1 if (i,j,k) is valid, 0 otherwise.
    # A rank-r decomposition means: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    #
    # We can try to use the following structured approach:
    # For each j in {0,1,2}, the terms A[i,j]*B[j,k] for i in {0,1}, k in {0..6}
    # form a 2x7 grid. If we can find a rank-1 decomposition of each 2x7 grid
    # that is consistent across j, we can reduce the rank.
    #
    # More specifically, we can try:
    # U[i,j] = alpha_i * beta_j
    # V[j,k] = gamma_j * delta_k
    # W[k,i] = epsilon_k * zeta_i
    # This gives: sum_r alpha_i[r] * beta_j[r] * gamma_j[r] * delta_k[r] * epsilon_k[r] * zeta_i[r]
    #
    # This is a rank-1 decomposition of the full tensor, which won't work.
    # Instead, let's try a more general approach.

    # Strategy: Use a randomized local search starting from the schoolbook
    # decomposition, trying to merge columns.

    def try_improve(rank, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using local search."""
        random.seed(seed)
        entries = [-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1]

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None

            # Initialize with a structured approach
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]

            # Assign each term to a column
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = random.randint(0, rank - 1)
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1

            # Now try to adjust entries to satisfy constraints
            for _ in range(100):
                if time.time() > deadline - 10.0:
                    break

                # Pick a random row and column, adjust
                if random.random() < 0.33:
                    a = random.randint(0, n * m - 1)
                    r = random.randint(0, rank - 1)
                    U[a][r] = random.choice(entries)
                elif random.random() < 0.5:
                    b = random.randint(0, m * p - 1)
                    r = random.randint(0, rank - 1)
                    V[b][r] = random.choice(entries)
                else:
                    c = random.randint(0, p * n - 1)
                    r = random.randint(0, rank - 1)
                    W[c][r] = random.choice(entries)

                # Check if valid (use float verification for speed)
                if verify_float(U, V, W, n, m, p):
                    # Verify with exact arithmetic
                    if verify_exact(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}

        return None

    # Try ranks from 41 down to 20
    for r in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 10.0:
            break
        result = try_improve(r, max_iters=200, seed=r * 7919)
        if result is not None:
            best = result
            best_rank = r
            break

    # If we found something better, try to go even lower
    if best_rank < n * m * p:
        for r in range(best_rank - 1, 15, -1):
            if time.time() > deadline - 10.0:
                break
            result = try_improve(r, max_iters=100, seed=r * 104729)
            if result is not None:
                best = result
                best_rank = r
                break

    # Final verification
    if verify_float(best["U"], best["V"], best["W"], n, m, p):
        if verify_exact(best["U"], best["V"], best["W"], n, m, p):
            return best
    else:
        # Fallback to schoolbook
        return schoolbook()

# EVOLVE-BLOCK-END
