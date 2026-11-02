"""Refined search for rank-minimal exact tensor decomposition of 3x5x6 matmul."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Schoolbook baseline
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)

    # Search for better decompositions
    best_result = _search_decomposition(n, m, p, deadline, best_rank)

    return best_result


def _schoolbook(n, m, p, rank):
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


def _verify(n, m, p, U, V, W, r):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_fast(n, m, p, U, V, W, r):
    """Faster verification using numpy."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all (i,j,k)

    # For each (i,j), compute the vector over k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            u_val = U_np[i * m + j, :]  # shape (r,)
            # For each k, we need: sum_c u_val[c] * V_np[j*p+k, c] * W_np[k*n+i, c]
            # This is: sum_c (u_val[c] * V_np[j*p+k, c]) * W_np[k*n+i, c]
            # Let's compute for all k at once
            # V_row = V_np[j*p:(j+1)*p, :]  # shape (p, r)
            # W_col = W_np[:, i]  # shape (p*n,) but we need W_np[k*n+i, :] for each k
            # Actually W_np[k*n+i, :] for k in range(p) gives us p vectors of shape (r,)
            # Let's stack them
            w_vecs = np.array([W_np[k * n + i, :] for k in range(p)])  # shape (p, r)
            v_vecs = V_np[j * p:(j + 1) * p, :]  # shape (p, r)
            # For each k: sum_c u_val[c] * v_vecs[k,c] * w_vecs[k,c]
            # = sum_c (u_val * v_vecs[k]) * w_vecs[k]
            # = dot(u_val * v_vecs[k], w_vecs[k])
            # For all k: this is element-wise product then sum over c
            # result[k] = sum_c u_val[c] * v_vecs[k,c] * w_vecs[k,c]
            result = np.sum(u_val[np.newaxis, :] * v_vecs * w_vecs, axis=1)  # shape (p,)
            if np.any(np.abs(result - 1.0) > 1e-9):
                return False
    return True


def _search_decomposition(n, m, p, deadline, current_best_rank):
    """Search for a better decomposition."""
    best_rank = current_best_rank
    best_result = _schoolbook(n, m, p, best_rank)

    # Strategy: Try structured decompositions based on block factorization
    # and random search with verification.

    # Try ranks from a reasonable lower bound up to current_best_rank - 1
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But we can do better. Let's try from 30 upward.

    start_rank = max(30, current_best_rank - 30)
    end_rank = current_best_rank - 1

    for r in range(start_rank, min(end_rank + 1, start_rank + 25)):
        if time.time() > deadline:
            break

        result = _try_rank_structured(n, m, p, r, deadline)
        if result is not None:
            best_rank = r
            best_result = result
            break

    # Also try some random searches for higher ranks
    for r in range(current_best_rank - 5, current_best_rank):
        if time.time() > deadline:
            break
        result = _try_rank_random(n, m, p, r, deadline)
        if result is not None:
            best_rank = r
            best_result = result
            break

    return best_result


def _try_rank_structured(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using structured patterns."""
    # Use block decomposition: split m=5 into 2+3
    # Try to find good decompositions for 3x2x6 and 3x3x6 blocks

    # First, try a simple block approach
    # Block 1: rows 0-2, cols 0-1 (3x2x6)
    # Block 2: rows 0-2, cols 2-4 (3x3x6)

    # For 3x2x6: schoolbook rank is 36
    # For 3x3x6: schoolbook rank is 54
    # Total: 90

    # Can we do better? Let's try to find decompositions with smaller ranks.

    # Approach: Use the fact that we can share some rank-1 terms between blocks
    # if the structures are compatible.

    # For now, let's try a direct random search with structured initialization.
    for trial in range(30):
        if time.time() > deadline:
            return None

        U, V, W = _generate_structured_candidate(n, m, p, r, trial)
        if _verify_fast(n, m, p, U, V, W, r):
            return {"U": U, "V": V, "W": W}

    return None


def _try_rank_random(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using random search."""
    for trial in range(20):
        if time.time() > deadline:
            return None

        U, V, W = _generate_random_candidate(n, m, p, r, trial)
        if _verify_fast(n, m, p, U, V, W, r):
            return {"U": U, "V": V, "W": W}

    return None


def _generate_structured_candidate(n, m, p, r, seed):
    """Generate a structured candidate decomposition."""
    random.seed(seed * 1000 + n * 100 + m * 10 + p)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Initialize with sparse random patterns
    for c in range(r):
        for idx in range(n * m):
            if random.random() < 0.5:
                U[idx][c] = random.choice([1, -1, 0.5, -0.5])

        for idx in range(m * p):
            if random.random() < 0.5:
                V[idx][c] = random.choice([1, -1, 0.5, -0.5])

        for idx in range(p * n):
            if random.random() < 0.5:
                W[idx][c] = random.choice([1, -1, 0.5, -0.5])

    return U, V, W


def _generate_random_candidate(n, m, p, r, seed):
    """Generate a random candidate decomposition."""
    random.seed(seed * 7 + n * 100 + m * 10 + p)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for c in range(r):
        for idx in range(n * m):
            if random.random() < 0.6:
                U[idx][c] = random.choice([1, -1, 0.5, -0.5, 2, -2])

        for idx in range(m * p):
            if random.random() < 0.6:
                V[idx][c] = random.choice([1, -1, 0.5, -0.5, 2, -2])

        for idx in range(p * n):
            if random.random() < 0.6:
                W[idx][c] = random.choice([1, -1, 0.5, -0.5, 2, -2])

    return U, V, W
# EVOLVE-BLOCK-END
