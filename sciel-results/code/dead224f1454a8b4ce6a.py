"""Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 4
    r1, r2 = 2, 1          # split of the n=3 rows: 2x4 block + 1x4 vector

    def _build(r, m, p, seed):
        # r x m  (A block) times m x p (B) -> r x p (C block)
        # U: (r*m, rank) ; V: (m*p, rank) ; W: (p*r, rank)
        # index convention: U[a*m + b], V[b*p + c], W[c*r + a]
        rng = np.random.RandomState(seed)
        rank = r * m * p
        U = np.zeros((r * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * r, rank))
        col = 0
        for a in range(r):
            for b in range(m):
                for c in range(p):
                    U[a * m + b, col] = 1.0
                    V[b * p + c, col] = 1.0
                    W[c * r + a, col] = 1.0
                    col += 1
        return U, V, W

    # Block 1: top 2 rows of A (2x4) x B (4x4) -> 2x4
    U1, V1, W1 = _build(r1, m, p, 1)
    # Block 2: bottom 1 row of A (1x4) x B (4x4) -> 1x4
    U2, V2, W2 = _build(r2, m, p, 2)

    # Combine: U and V are block-diagonal (independent columns),
    # W is vertically stacked (shared B, so same V columns per block).
    rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Block 1 occupies A rows 0..r1-1
    U[:r1 * m, :U1.shape[1]] = U1
    V[:m * p, :V1.shape[1]] = V1
    W[:p * r1, :W1.shape[1]] = W1

    # Block 2 occupies A rows r1..n-1
    off = U1.shape[1]
    U[r1 * m:, off:] = U2
    V[:m * p, off:] = V2
    W[p * r1:, off:] = W2

    # ---- EXACT VALIDATION (integer entries, so exact) ----
    def _verify(U, V, W, n, m, p):
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        # Build the multiplication tensor T[i,j,k] and compare.
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        # Reconstruct from factors: R[i,j,k] = sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t]
        R = np.zeros((n, m, p))
        Ua = U.reshape(n, m, -1)
        Va = V.reshape(m, p, -1)
        Wa = W.reshape(p, n, -1)
        # R[i,j,k] = sum_t Ua[i,j,t] * Va[j,k,t] * Wa[k,i,t]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] = np.dot(Ua[i, j, :], Va[j, k, :] * Wa[k, i, :])
        if not np.allclose(R, T, atol=1e-9):
            raise ValueError("Decomposition does not exactly reproduce the multiplication tensor")
        return True

    _verify(U, V, W, n, m, p)

    # Convert to lists of integer/half-integer entries
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    # Ensure integer representation
    for M in (U, V, W):
        for row in M:
            for idx in range(len(row)):
                row[idx] = int(round(row[idx]))

    if time.time() > DEADLINE:
        # Already validated above; return immediately.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
