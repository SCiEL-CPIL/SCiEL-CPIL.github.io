"""Rewritten program: rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0

def _check(U, V, W, n, m, p):
    """Verify exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns U (4x7), V (4x7), W (4x7) with indices:
      U: (i,j) -> i*2+j
      V: (j,k) -> j*2+k
      W: (k,i) -> k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A11 - A22)(B11 + B22)
    U[0,0] = 1;  U[3,0] = -1
    V[0,0] = 1;  V[3,0] = 1
    W[0,0] = 1;  W[3,0] = 1   # C11 and C22

    # M2 = (A21 + A22) B11
    U[2,1] = 1;  U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1                 # C21

    # M3 = A11 (B12 - B22)
    U[0,2] = 1
    V[1,2] = 1;  V[3,2] = -1
    W[1,2] = 1                 # C12

    # M4 = A22 (B21 - B11)
    U[3,3] = 1
    V[2,3] = 1;  V[0,3] = -1
    W[0,3] = 1;  W[2,3] = 1   # C11 and C21

    # M5 = (A11 + A12) B22
    U[0,4] = 1;  U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1;  W[1,4] = 1  # C11 (neg) and C12

    # M6 = (A21 - A11)(B11 + B12)
    U[2,5] = 1;  U[0,5] = -1
    V[0,5] = 1;  V[1,5] = 1
    W[3,5] = 1                 # C22

    # M7 = (A12 - A22)(B21 + B22)
    U[1,6] = 1;  U[3,6] = -1
    V[2,6] = 1;  V[3,6] = 1
    W[0,6] = 1                 # C11

    return U, V, W

def _hybrid_18():
    """
    Build a rank-18 decomposition:
      - Strassen (rank 7) for C[:, 0:2] = A * B[:, 0:2]
      - Strassen (rank 7) for C[:, 2:4] = A * B[:, 2:4]
      - Schoolbook (rank 4) for C[:, 4]  = A * B[:, 4]
    """
    n, m, p = 2, 2, 5
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    U1, V1, W1 = _strassen_2x2()

    # Part 1: columns 0,1 of B and C
    for c in range(7):
        U[:, c] = U1[:, c]
        # V1 index r = j*2 + k_local (k_local in {0,1}); global V index = j*p + k_local
        for r in range(4):
            j = r // 2
            k_local = r % 2
            V[j * p + k_local, c] = V1[r, c]
        # W1 index r = k_local*2 + i; global W index = (k_local)*n + i = same
        for r in range(4):
            k_local = r // 2
            i = r % 2
            W[k_local * n + i, c] = W1[r, c]

    # Part 2: columns 2,3 of B and C
    off = 7
    for c in range(7):
        U[:, off + c] = U1[:, c]
        for r in range(4):
            j = r // 2
            k_local = r % 2
            V[j * p + (2 + k_local), off + c] = V1[r, c]
        for r in range(4):
            k_local = r // 2
            i = r % 2
            W[(2 + k_local) * n + i, off + c] = W1[r, c]

    # Part 3: column 4 of B and C (schoolbook, rank 4)
    off = 14
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        U[i * m + j, off + idx] = 1
        V[j * p + 4, off + idx] = 1
        W[4 * n + i, off + idx] = 1

    return U, V, W

def _try_rank(n, m, p, rank, time_budget):
    """
    Attempt to find a rank-r decomposition via structured search.
    Strategy: for each column, pick a small support pattern and small integer values,
    then verify. Also try to mutate the hybrid-18 and prune columns.
    """
    deadline = time.time() + time_budget
    U0, V0, W0 = _hybrid_18()

    # --- Attempt 1: prune columns from the rank-18 hybrid ---
    # Remove one column at a time and see if the remaining 17 still work.
    for r in range(rank, 12, -1):
        if time.time() > deadline:
            break
        # Try removing columns from the 18-column hybrid
        if r >= 17:
            for remove_col in range(18):
                if time.time() > deadline:
                    break
                U_try = np.delete(U0, remove_col, axis=1)
                V_try = np.delete(V0, remove_col, axis=1)
                W_try = np.delete(W0, remove_col, axis=1)
                if _check(U_try, V_try, W_try, n, m, p):
                    return U_try.tolist(), V_try.tolist(), W_try.tolist()
        if r == 16:
            # Try removing two columns
            for c1 in range(18):
                if time.time() > deadline:
                    break
                for c2 in range(c1 + 1, 18):
                    U_try = np.delete(U0, [c1, c2], axis=1)
                    V_try = np.delete(V0, [c1, c2], axis=1)
                    W_try = np.delete(W0, [c1, c2], axis=1)
                    if _check(U_try, V_try, W_try, n, m, p):
                        return U_try.tolist(), V_try.tolist(), W_try.tolist()

    # --- Attempt 2: random sparse search for the target rank ---
    candidates = [-2, -1, 1, 2]
    np.random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        attempts += 1
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        ok = True
        for c in range(rank):
            nu = np.random.randint(1, 4)
            nv = np.random.randint(1, 4)
            nw = np.random.randint(1, 4)
            ui = np.random.choice(n*m, size=nu, replace=False)
            vi = np.random.choice(m*p, size=nv, replace=False)
            wi = np.random.choice(p*n, size=nw, replace=False)
            U[ui, c] = np.random.choice(candidates, size=nu)
            V[vi, c] = np.random.choice(candidates, size=nv)
            W[wi, c] = np.random.choice(candidates, size=nw)
        if _check(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()

    return None

def construct():
    n, m, p = DIMENSIONS

    # Start with the rank-18 hybrid
    U, V, W = _hybrid_18()
    if not _check(U, V, W, n, m, p):
        # Fallback: schoolbook rank 20
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 18
    best = (U.tolist(), V.tolist(), W.tolist())

    # Try to find a lower rank
    remaining = _DEADLINE - (time.time() - _START) - _SAFETY
    for r in range(17, 12, -1):
        if remaining < 2.0:
            break
        budget = min(remaining - 1.0, 20.0)
        result = _try_rank(n, m, p, r, budget)
        if result is not None:
            best = result
            best_rank = r
            break
        remaining = _DEADLINE - (time.time() - _START) - _SAFETY

    U_out, V_out, W_out = best
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
