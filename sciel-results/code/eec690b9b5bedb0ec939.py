"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 3]

def _deadline_guard(seconds=140.0):
    return time.time() + seconds

def _verify(U, V, W, n, m, p, rank):
    """Exact integer verification of the tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for all i,j,k."""
    # build U,V,W as lists of lists of integers (scale by 2 to clear halves)
    # We'll work in doubled space: entries are 2*x where x in {0, +/-0.5, +/-1, ...}
    U2 = [[2 * U[i][c] for c in range(rank)] for i in range(n * m)]
    V2 = [[2 * V[j][c] for c in range(rank)] for j in range(m * p)]
    W2 = [[2 * W[k][c] for c in range(rank)] for k in range(p * n)]
    # T[i,j,k] = sum_c U2[i*m+j,c]*V2[j*p+k,c]*W2[k*n+i,c] / 8  must equal 1
    # => sum_c ... == 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:
                    return False
    return True

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-START
def construct():
    deadline = _deadline_guard(140.0)
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Candidate entry values: 0, +/-1/2, +/-1, +/-3/2, +/-2
    entries = [0, 1, -1, 0.5, -0.5, 1.5, -1.5, 2, -2]

    def _try_random(rank, trials, seed):
        nonlocal best, best_rank
        rng = random.Random(seed)
        for t in range(trials):
            if time.time() > deadline:
                return
            # random sparse matrices
            U = [[rng.choice(entries) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice(entries) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice(entries) for _ in range(rank)] for _ in range(p * n)]
            if _verify(U, V, W, n, m, p, rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                return True
        return False

    # Try ranks from best_rank-1 down to a lower bound
    # Lower bound for 2x3x3: known to be 15 (de Groote 1978). Try 15..17.
    for rank in range(best_rank - 1, 14, -1):
        if time.time() > deadline:
            break
        # a few randomized trials
        if _try_random(rank, 2000, rank * 1000 + 7):
            if rank <= 15:
                break

    # If we found rank 15, that's the known optimum; stop.
    if best_rank <= 15:
        return best

    # Fallback: return best found (likely schoolbook 18)
    return best
# EVOLVE-BLOCK-END
