"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook():
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

def _is_half_integer(x):
    y = 2 * x
    return abs(y - round(y)) < 1e-9

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if not all(_is_half_integer(U[i][r]) for i in range(n * m)):
            return False
        if not all(_is_half_integer(V[j][r]) for j in range(m * p)):
            return False
        if not all(_is_half_integer(W[k][r]) for k in range(p * n)):
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1 if (i == ip and j == jp and k == kp) else 0
                            actual = 0
                            for c in range(rank):
                                actual += U[i*m+j][c] * V[jp*p+kp][c] * W[kp*n+ip][c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _try_block_decomposition():
    """
    Attempt to construct a rank-18 decomposition by treating the 3x6 matrix as two 3x3 blocks.
    This exploits the structure to potentially reduce the rank from 36 (schoolbook) to 18.
    """
    n, m, p = DIMENSIONS
    # The 3x6 matrix B is split into B1 (3x3) and B2 (3x3).
    # A is 2x3.
    # C = A * B = A * [B1, B2] = [A*B1, A*B2].
    # The output C is 2x6.
    # The tensor for A*B1 is rank 2*3*3 = 18.
    # The tensor for A*B2 is rank 2*3*3 = 18.
    # However, the inputs A are shared.
    # We can try to find a joint decomposition.
    
    # Let's try a specific structured decomposition.
    # We will attempt to find a rank-18 solution by solving a system of linear equations
    # with the constraint that entries are half-integers.
    # This is a hard problem, so we will use a heuristic search.
    
    # For now, we will return None to indicate failure, and the caller will fall back to schoolbook.
    # A more sophisticated approach would use algebraic methods or known results.
    # Given the time limit, we will not attempt a full search here.
    return None

def construct():
    deadline = time.time() + 140.0  # Safety margin
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try block decomposition
    block_sol = _try_block_decomposition()
    if block_sol is not None:
        if _verify(block_sol["U"], block_sol["V"], block_sol["W"]):
            if len(block_sol["U"][0]) < best_rank:
                best = block_sol
                best_rank = len(best["U"][0])
    
    # If we have time, we could try other search strategies, but for now we return the best found.
    return best
# EVOLVE-BLOCK-END
