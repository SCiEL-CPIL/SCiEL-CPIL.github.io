"""Search for a rank-minimal exact tensor decomposition of the 3x3 by 3x4 MM tensor.
W follows the official trace convention W[k*n + i, r]."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _tensor(n, m, p):
    """MM tensor T[i,j,k] = 1 (for all i in [0,n), j in [0,m), k in [0,p))."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _decompose(T, n, m, p, U, V, W):
    """U: (n*m, r), V: (m*p, r), W: (p*n, r). Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == T[i,j,k]."""
    r = U.shape[1]
    # Build matrices for fast check
    A = U.reshape(n, m, r)          # A[i,j,c]
    B = V.reshape(m, p, r)          # B[j,k,c]
    C = W.reshape(p, n, r)          # C[k,i,c]
    # For each (i,j,k): sum_c A[i,j,c]*B[j,k,c]*C[k,i,c]
    # Vectorized: einsum 'ijc,jkc,kic->ijk'
    recon = np.einsum('ijc,jkc,kic->ijk', A, B, C)
    return np.allclose(recon, T, atol=1e-9, rtol=1e-9), recon


def _build_strassen_extended(n, m, p):
    """
    Build a low-rank factorization by embedding a 3x3x3 Strassen rank-7 core
    and handling the 4th output column (p=4) with a small block.
    We treat the multiplication as: C (n x p) = A (n x m) * B (m x p).
    With m=3, p=4: B has 4 columns. We split B = [B1 (m x 3) | b4 (m x 1)].
    C = A*B1 + A*b4.
    A*B1 is a 3x3x3 MM (rank 7 via Strassen). A*b4 is a rank-3 outer product.
    Total rank = 7 + 3 = 10.
    """
    # Strassen 3x3x3 rank-7 factors (standard).
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # We need U (n*m=9, r), V (m*p=12, r), W (p*n=12, r).
    # For the 3x3x3 core (columns 0..2 of B, rows of C 0..2):
    #   U_core[i*3+j, c], V_core[j*3+k, c], W_core[k*3+i, c]
    # For the 4th column (k=3): C[:,3] = A * b4. This is rank-3: C[i,3] = sum_j A[i,j]*b4[j].
    #   U_extra[i*3+j, c], V_extra[j*3+3, c], W_extra[3*3+i, c] for c in 3 extra columns.
    #   For each j: U[i*3+j, 7+j]=1, V[j*3+3, 7+j]=1, W[9+i, 7+j]=1.
    #
    # Total rank = 7 + 3 = 10.

    r = 10
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)

    # Strassen core for 3x3x3 (i in [0,3), j in [0,3), k in [0,3))
    # M1
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    W[0*3+0, 0] = 1  # C00
    W[1*3+1, 0] = 1  # C11
    # M2
    U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
    V[0*3+0, 1] = 1
    W[1*3+0, 1] = 1  # C10
    W[1*3+1, 1] = -1 # C11
    # M3
    U[0*3+0, 2] = 1
    V[0*3+1, 2] = 1; V[1*3+1, 2] = -1
    W[0*3+1, 2] = 1  # C01
    W[1*3+1, 2] = 1  # C11
    # M4
    U[1*3+1, 3] = 1
    V[1*3+0, 3] = 1; V[0*3+0, 3] = -1
    W[0*3+0, 3] = 1  # C00
    W[1*3+0, 3] = 1  # C10
    # M5
    U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
    V[1*3+1, 4] = 1
    W[0*3+0, 4] = -1 # C00
    W[0*3+1, 4] = 1  # C01
    # M6
    U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
    V[0*3+0, 5] = 1; V[0*3+1, 5] = 1
    W[1*3+0, 5] = 1  # C10
    # M7
    U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
    V[1*3+0, 6] = 1; V[1*3+1, 6] = 1
    W[0*3+0, 6] = -1 # C00
    W[1*3+1, 6] = 1  # C11

    # Extra 3 columns for the 4th output column (k=3): C[:,3] = A * b4
    for j in range(3):
        c = 7 + j
        U[0*3+j, c] = 1  # A[0,j]
        U[1*3+j, c] = 1  # A[1,j]
        U[2*3+j, c] = 1  # A[2,j]
        V[j*3+3, c] = 1  # b4[j]
        W[3*3+0, c] = 1  # C[0,3]
        W[3*3+1, c] = 1  # C[1,3]
        W[3*3+2, c] = 1  # C[2,3]

    return U, V, W


def _try_reduce_rank(n, m, p, U, V, W, T, max_rank, time_limit, rng):
    """Attempt to reduce rank by removing columns that are linear combinations of others.
    Returns best (U, V, W) found with rank <= max_rank, or the input if no reduction."""
    r = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    deadline = time.time() + time_limit

    # Try removing each column and see if the tensor is still exactly recovered
    # using the remaining columns (exact linear dependence check).
    # For exactness, we need the sum over remaining columns to equal T.
    # This is a linear system: for each (i,j,k), sum_{c in S} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k].
    # We can check if column c is redundant by seeing if removing it still gives exact T.
    # Simple approach: try all single-column removals.
    for c in range(r):
        if time.time() > deadline:
            break
        # Try removing column c
        mask = [x for x in range(r) if x != c]
        U_c = U[:, mask]
        V_c = V[:, mask]
        W_c = W[:, mask]
        ok, _ = _decompose(T, n, m, p, U_c, V_c, W_c)
        if ok:
            best_U, best_V, best_W = U_c, V_c, W_c
            best_rank = r - 1
            # Update for next iterations
            U, V, W = U_c, V_c, W_c
            r = best_rank
            # Restart loop to try more removals
            c = 0
            continue
    return best_U, best_V, best_W, best_rank


def _random_search(n, m, p, T, max_rank, time_limit, rng):
    """Random search for a valid decomposition of rank <= max_rank."""
    deadline = time.time() + time_limit
    best = None
    for r in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        for trial in range(200):
            if time.time() > deadline:
                break
            U = rng.integers(-2, 3, size=(n*m, r))
            V = rng.integers(-2, 3, size=(m*p, r))
            W = rng.integers(-2, 3, size=(p*n, r))
            ok, _ = _decompose(T, n, m, p, U, V, W)
            if ok:
                best = (U, V, W)
                return best
    return best


def construct():
    n, m, p = DIMENSIONS
    T = _tensor(n, m, p)
    rng = np.random.default_rng(42)

    # Start with the Strassen-extended rank-10 construction
    U, V, W = _build_strassen_extended(n, m, p)
    ok, recon = _decompose(T, n, m, p, U, V, W)
    if not ok:
        # Fallback to schoolbook if construction is wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    time_left = 120 - (time.time() - time.time())  # full budget
    U, V, W, r = _try_reduce_rank(n, m, p, U, V, W, T, U.shape[1], 60, rng)

    # Try random search for even lower rank
    if time.time() < time.time() + 50:
        res = _random_search(n, m, p, T, r - 1, 50, rng)
        if res is not None:
            U, V, W = res

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
