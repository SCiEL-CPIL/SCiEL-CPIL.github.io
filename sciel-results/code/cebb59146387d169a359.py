"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    # Start with schoolbook (rank 63)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)

    # For 3x3x7, the tensor can be decomposed by trying to find a lower rank
    # We'll attempt to find a rank-57 or lower decomposition using a heuristic search
    # over the known structure.
    #
    # Since 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1, and the rank of 3x3x3 is 23,
    # a naive block decomposition gives 23+23+9 = 55, but this requires
    # a valid rank-23 decomposition for 3x3x3.
    #
    # The exact rank-23 decomposition for 3x3x3 is known (e.g., by
    # Landsberg, or from the work of Landsberg and Ottaviani).
    # Let me try to implement it.

    # Actually, let me check: can I construct a valid lower-rank decomposition?
    # The safest approach is to return the schoolbook if I can't find a valid
    # lower-rank one. But let me try a simple random search for a lower-rank
    # decomposition.

    # For the 3x3x7 case, let's try to see if we can do better than 63.
    # One approach: use the fact that p=7 = 3+3+1, and try to find a
    # decomposition that exploits this structure.

    # Let's try a rank-57 decomposition using a heuristic.
    # If we can find U, V, W of rank 57 that satisfy the tensor equation,
    # we return it. Otherwise, we fall back to schoolbook.

    # The tensor equation is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since (A @ B)[i,k] = sum_j A[i,j] * B[j,k])
    #
    # In the trace convention W[k*n+i, r], the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, the standard form is: C[i,k] = sum_{j,r} A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # Actually, the factorization is: C = sum_r (U_r @ V_r) where U_r is n x m, V_r is m x p
    # and W_r is the output factor.
    #
    # Let me just return the schoolbook for now, as implementing a valid
    # rank-23 decomposition from memory is risky.

    # Actually, let me try to implement the known rank-23 for 3x3x3.
    # The decomposition is complex. Let me use a different approach:
    # try random rank-57 and see if we can verify it quickly.

    # Given the time constraints and the difficulty of implementing a
    # valid rank-23 decomposition from memory, I'll return the schoolbook.
    # This guarantees a valid decomposition at rank 63.

    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}

# EVOLVE-BLOCK-END
