import numpy as np
import time
from scipy.linalg import solve

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        # Fallback: schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END
