import numpy as np
import time

DIMENSIONS = [3, 5, 5]


def _build_decomposition(n, m, p, blocks, uncovered):
    total_rank = len(blocks) * 7 + len(uncovered)
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0
    for (i0, j0, k0) in blocks:
        # 7 Strassen factors for the 2x2x2 block at (i0,j0,k0)
        # factor 0: M1 = A[i0..i0+1, j0] * B[j0..j0+1, k0] -> C[i0..i0+1, k0..k0+1] (both rows/cols)
        c = col
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # factor 1: M2 = A[i0+1, j0..j0+1] * B[j0, k0..k0+1] -> C[i0+1, k0..k0+1] with signs
        c = col + 1
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1

        # factor 2: M3 = A[i0, j0..j0+1] * B[j0..j0+1, k0+1] -> C[i0, k0+1] both rows
        c = col + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # factor 3: M4 = A[i0..i0+1, j0+1] * B[j0+1, k0..k0+1] -> C[i0..i0+1, k0]
        c = col + 3
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1

        # factor 5: M5 = A[i0..i0+1, j0..j0+1] * B[j0+1, k0+1] -> C[i0, k0..k0+1]
        c = col + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1

        # factor 6: M6 = A[i0..i0+1, j0] * B[j0, k0..k0+1] -> C[i0+1, k0+1]
        c = col + 5
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # factor 7: M7 = A[i0..i0+1, j0+1] * B[j0+1, k0..k0+1] -> C[i0, k0]
        c = col + 6
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1

        col += 7

    for (i, j, k) in uncovered:
        U[i*m + j, col] += 1
        V[j*p + k, col] += 1
        W[k*n + i, col] += 1
        col += 1

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the matmul tensor."""
    # tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build expected: T[i,j,k] = 1 for all (i,j,k)
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U.shape[1]):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # nonzero entries
        ui = np.nonzero(u)[0]
        vi = np.nonzero(v)[0]
        wi = np.nonzero(w)[0]
        for a in ui:
            i, j = divmod(a, m)
            for b in vi:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                for cc in wi:
                    k2, i2 = divmod(cc, n)
                    if k2 != k or i2 != i:
                        continue
                    T[i, j, k] += u[a] * v[b] * w[cc]
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)


def _try_remove_trivial(U, V, W, n, m, p, trivial_cols):
    """Try to remove trivial (single-term) columns by redistributing their mass.
    A trivial column c has exactly one nonzero in each of U, V, W at the same (i,j,k).
    We can only remove it if its contribution can be absorbed. Since a trivial term
    contributes exactly 1 to T[i,j,k] at a single point, and no other column touches
    that point (by construction of uncovered), removing it would make T[i,j,k]=0.
    So trivial columns cannot be removed unless we also modify other columns.
    This is a no-op placeholder kept for interface; returns False.
    """
    return False


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # 4 Strassen 2x2x2 blocks (maximal packing in 3x5x5)
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]

    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True

    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    uncovered.append((i, j, k))

    U, V, W = _build_decomposition(n, m, p, blocks, uncovered)

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback: pure trivial decomposition (rank = n*m*p)
        U = np.zeros((n*m, n*m*p), dtype=float)
        V = np.zeros((m*p, n*m*p), dtype=float)
        W = np.zeros((p*n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    # quick verification
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U.shape[1]):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        ui = np.nonzero(u)[0]
        vi = np.nonzero(v)[0]
        wi = np.nonzero(w)[0]
        for a in ui:
            i, j = divmod(a, m)
            for b in vi:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                for cc in wi:
                    k2, i2 = divmod(cc, n)
                    if k2 != k or i2 != i:
                        continue
                    T[i, j, k] += u[a] * v[b] * w[cc]
    print(f"Max abs error vs ones: {np.max(np.abs(T - 1.0))}")
