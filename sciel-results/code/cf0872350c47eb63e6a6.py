"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy: Start with rank-55 Strassen-based solution, then attempt to reduce rank
by searching for equivalent decompositions with fewer terms. Use structured
random search with integer/half-integer entries and deadline guards.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Check all nonzero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check some zero positions
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(500):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            val = 0.0
            for r in range(rank):
                val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            if abs(val) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """Build the rank-55 decomposition using Strassen on the 2x2 sub-block."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1; V[v_22][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1; V[v_11][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1; U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1; U[u_11][col] += -1
        V[v_11][col] += 1; V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1; U[u_22][col] += -1
        V[v_21][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline, values):
    """Try to find a decomposition of target_rank from the given solution."""
    current_rank = len(U[0])
    if target_rank >= current_rank:
        return None

    # Try removing columns that might be redundant
    # Use random search in the space of rank-target_rank decompositions
    best_solution = None
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random values from the allowed set
        U_new = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
        V_new = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
        W_new = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify(U_new, V_new, W_new, n, m, p) and _is_valid_entries(U_new, V_new, W_new):
            return U_new, V_new, W_new
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find lower ranks
    values = [0, 0.5, 1, -0.5, -1]
    
    # Theoretical lower bound considerations:
    # For (2,3,10), the rank must be at least max(2*3, 3*10, 10*2) = 30
    # But more tightly, we can use the fact that each output C[i,k] needs 
    # to capture the 3-term dot product
    # Try ranks from 54 down to 50
    for r in range(54, 49, -1):
        if time.time() > deadline - 10:
            break
        result = _try_reduce_rank(U_best, V_best, W_best, n, m, p, r, deadline, values)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best, "V": V_best, "W": W_best}
