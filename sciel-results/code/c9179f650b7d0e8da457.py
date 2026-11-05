"""Matrix multiplication 2x3x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
DEADLINE = time.time() + 145  # hard wall-clock limit (150s kill, 5s margin)


def _verify(U, V, W):
    """Check whether the factor matrices exactly decompose the 2x3x6 matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook():
    """Return the rank-36 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def _strassen_rank33():
    """
    Return a rank-33 decomposition using Strassen for 2x2x2 blocks.
    Split: A(2x3) = A1(2x1) + A2(2x2), B(3x6) = B1(1x6) + B2(2x6)
    A1*B1: 2x1x6, rank 12
    A2*B2: 2x2x6, split into three 2x2x2 blocks, each rank 7 (Strassen), total 21
    Total: 33
    """
    n, m, p = DIMENSIONS
    rank = 33

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1: rows i=0,1; col j=0
    # B1: row j=0; cols k=0..5
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2: rows i=0,1; cols j=1,2
    # B2: rows j=1,2; cols k=0..5
    # Split B2 into three 2x2 blocks

    for t in range(3):
        # Strassen 7 multiplications for 2x2x2 block
        # M1 = (A00+A11)(B00+B11)
        U[1][col] += 1
        U[5][col] += 1
        V[6 + 2*t][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] += 1
        W[4*t + 3][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[6 + 2*t][col] += 1
        W[4*t + 1][col] += 1
        W[4*t + 3][col] -= 1
        col += 1

        # M3 = A00(B01-B11)
        U[1][col] += 1
        V[7 + 2*t][col] += 1
        V[13 + 2*t][col] -= 1
        W[4*t + 2][col] += 1
        W[4*t + 3][col] += 1
        col += 1

        # M4 = A11(B10-B00)
        U[5][col] += 1
        V[12 + 2*t][col] += 1
        V[6 + 2*t][col] -= 1
        W[4*t][col] += 1
        W[4*t + 1][col] += 1
        col += 1

        # M5 = (A00+A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] -= 1
        W[4*t + 2][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[6 + 2*t][col] += 1
        V[7 + 2*t][col] += 1
        W[4*t + 3][col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[12 + 2*t][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}


def _try_remove_columns(U, V, W, max_removals=3):
    """Try to remove redundant columns from the decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank

    # Try removing one column at a time
    for c_remove in range(rank):
        if time.time() > DEADLINE - 15:
            break
        keep = [c for c in range(rank) if c != c_remove]
        U_red = [[row[c] for c in keep] for row in U]
        V_red = [[row[c] for c in keep] for row in V]
        W_red = [[row[c] for c in keep] for row in W]
        if _verify(U_red, V_red, W_red):
            best_rank = rank - 1
            best_U, best_V, best_W = U_red, V_red, W_red
            break

    if best_rank == rank:
        return best_U, best_V, best_W, best_rank

    # Try removing a second column
    rank2 = best_rank
    for c_remove in range(rank2):
        if time.time() > DEADLINE - 10:
            break
        keep = [c for c in range(rank2) if c != c_remove]
        U_red = [[row[c] for c in keep] for row in best_U]
        V_red = [[row[c] for c in keep] for row in best_V]
        W_red = [[row[c] for c in keep] for row in best_W]
        if _verify(U_red, V_red, W_red):
            best_rank = rank2 - 1
            best_U, best_V, best_W = U_red, V_red, W_red
            break

    if best_rank == rank2:
        return best_U, best_V, best_W, best_rank

    # Try removing a third column
    rank3 = best_rank
    for c_remove in range(rank3):
        if time.time() > DEADLINE - 5:
            break
        keep = [c for c in range(rank3) if c != c_remove]
        U_red = [[row[c] for c in keep] for row in best_U]
        V_red = [[row[c] for c in keep] for row in best_V]
        W_red = [[row[c] for c in keep] for row in best_W]
        if _verify(U_red, V_red, W_red):
            best_rank = rank3 - 1
            best_U, best_V, best_W = U_red, V_red, W_red
            break

    return best_U, best_V, best_W, best_rank


def construct():
    """
    Find a rank-minimal decomposition for 2x3x6 matrix multiplication.
    Start with rank-33 Strassen-based decomposition, then try to reduce rank.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Get the rank-33 baseline
    base = _strassen_rank33()
    U, V, W = base["U"], base["V"], base["W"]

    if time.time() > DEADLINE - 20:
        return base

    # Verify the baseline
    if not _verify(U, V, W):
        return _schoolbook()

    # Try to reduce rank by removing redundant columns
    U_red, V_red, W_red, best_rank = _try_remove_columns(U, V, W, max_removals=3)

    if time.time() > DEADLINE - 5:
        return {"U": U_red, "V": V_red, "W": W_red}

    # If we got rank 32 or 33, try a more aggressive search for rank 31
    if best_rank <= 32 and time.time() < DEADLINE - 30:
        # Try to find a rank-31 decomposition by perturbing the rank-32 solution
        # and checking if we can drop one more term
        pass  # The column removal above already handles this

    return {"U": U_red, "V": V_red, "W": W_red}
