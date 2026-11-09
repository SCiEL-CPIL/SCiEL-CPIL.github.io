"""Matrix multiplication 3,3,3 - rank-minimal exact tensor decomposition.

We search for an exact rank-23 decomposition with integer (or half-integer)
entries.  The known optimal rank for 3x3 is 23.  We perform a structured search
using alternating minimization with multiple restarts and a final polish step.
If no exact rank-23 decomposition is found within the time budget, we fall back
to the schoolbook rank-27.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-8):
    """Check whether the factor matrices exactly decompose the matrix-multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for k_prime in range(p):
            for j in range(m):
                for j_prime in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k_prime * n + i][c] * U[i * m + j][c] * V[j_prime * p + k_prime][c]
                    expected = 1.0 if j == j_prime else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _alternating_minimize(U, V, W, n, m, p, rank, max_iter=50, deadline=None):
    """Perform alternating minimization to reduce the error."""
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Update U: fix V and W, solve for U
        A_rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k_prime in range(p):
                        row = [0.0] * (n * m * rank)
                        for c in range(rank):
                            row[(i * m + j) * rank + c] = W[k_prime * n + i][c] * V[j_prime * p + k_prime][c]
                        A_rows.append(row)
                        b_vec.append(1.0 if j == j_prime else 0.0)
        A_mat = np.array(A_rows)
        b_arr = np.array(b_vec)
        try:
            U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
        except Exception:
            break
        U = U_flat.reshape(n * m, rank)

        # Update V: fix U and W, solve for V
        A_rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k_prime in range(p):
                        row = [0.0] * (m * p * rank)
                        for c in range(rank):
                            row[(j_prime * p + k_prime) * rank + c] = W[k_prime * n + i][c] * U[i * m + j][c]
                        A_rows.append(row)
                        b_vec.append(1.0 if j == j_prime else 0.0)
        A_mat = np.array(A_rows)
        b_arr = np.array(b_vec)
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
        except Exception:
            break
        V = V_flat.reshape(m * p, rank)

        # Update W: fix U and V, solve for W
        A_rows = []
        b_vec = []
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k_prime in range(p):
                        row = [0.0] * (p * n * rank)
                        for c in range(rank):
                            row[(k_prime * n + i) * rank + c] = U[i * m + j][c] * V[j_prime * p + k_prime][c]
                        A_rows.append(row)
                        b_vec.append(1.0 if j == j_prime else 0.0)
        A_mat = np.array(A_rows)
        b_arr = np.array(b_vec)
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
        except Exception:
            break
        W = W_flat.reshape(p * n, rank)

    return U, V, W

def _try_round_and_verify(U, V, W, n, m, p, rank, deadline=None):
    """Try rounding to integers and half-integers, and verify."""
    # Try rounding to nearest half-integer
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    U_list = [[U_half[r, c] for c in range(rank)] for r in range(n * m)]
    V_list = [[V_half[r, c] for c in range(rank)] for r in range(m * p)]
    W_list = [[W_half[r, c] for c in range(rank)] for r in range(p * n)]
    if _verify(U_list, V_list, W_list):
        return U_list, V_list, W_list

    # Try rounding to nearest integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    U_list = [[U_int[r, c] for c in range(rank)] for r in range(n * m)]
    V_list = [[V_int[r, c] for c in range(rank)] for r in range(m * p)]
    W_list = [[W_int[r, c] for c in range(rank)] for r in range(p * n)]
    if _verify(U_list, V_list, W_list):
        return U_list, V_list, W_list

    return None

def _try_rank_23_search(max_attempts=100, deadline=110.0):
    """Search for a rank-23 exact decomposition using multiple strategies."""
    n, m, p = 3, 3, 3
    rank = 23

    # Strategy 1: Random initialization with alternating minimization
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        rng = random.Random(attempt * 7919 + 42)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for _ in range(n * m * rank):
            U.flat[_] = rng.choice([-2, -1, 0, 1, 2])
        for _ in range(m * p * rank):
            V.flat[_] = rng.choice([-2, -1, 0, 1, 2])
        for _ in range(p * n * rank):
            W.flat[_] = rng.choice([-2, -1, 0, 1, 2])

        U, V, W = _alternating_minimize(U, V, W, n, m, p, rank, max_iter=30, deadline=deadline - 5)
        result = _try_round_and_verify(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            return result

    # Strategy 2: Start from schoolbook and try to reduce rank
    # Create a rank-27 schoolbook solution
    U27 = np.zeros((n * m, 27))
    V27 = np.zeros((m * p, 27))
    W27 = np.zeros((p * n, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i * m + j, col] = 1
                V27[j * p + k, col] = 1
                W27[k * n + i, col] = 1
                col += 1

    # Try to find a rank-23 solution by randomly projecting
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        rng = np.random.RandomState(attempt * 104729 + 123)
        # Create a random 27x23 matrix and project
        P = rng.randn(27, rank)
        U = U27 @ P
        V = V27 @ P
        W = W27 @ P

        U, V, W = _alternating_minimize(U, V, W, n, m, p, rank, max_iter=30, deadline=deadline - 5)
        result = _try_round_and_verify(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            return result

    return None

def construct():
    deadline = time.time() + 115  # Leave 5s safety margin before 120s

    # Try to find a rank-23 decomposition
    result = _try_rank_23_search(max_attempts=50, deadline=deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 27
    n, m, p = DIMENSIONS
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
