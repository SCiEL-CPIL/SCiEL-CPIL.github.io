"""Matrix multiplication 3,5,6 rank-minimal decomposition.

Strategy:
  1. Build the full multiplication tensor T (size 15x30x18) with the official
     trace convention: T[i*m+j, j*p+k, k*n+i] = 1.
  2. Try a block-Kronecker construction: split the 5-dimension into 3+2 and
     the 6-dimension into 3+3, then use known low-rank decompositions for the
     resulting block products.
  3. If that fails, perform a structured search over ranks using randomized
     half-integer factors with a verification step.
  4. Fall back to the schoolbook decomposition (rank 90) if nothing better is
     found within the time budget.

The program keeps the construct() entry point and returns
{'U': ..., 'V': ..., 'W': ...} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Compute T_computed[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=1e-9)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _random_half_int_matrix(rows, cols, seed=None):
    """Generate a random matrix with entries in {-2,-1,-0.5,0,0.5,1,2}."""
    rng = np.random.default_rng(seed)
    vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    return rng.choice(vals, size=(rows, cols))


def _try_block_kronecker(n, m, p, deadline):
    """
    Try a block-based Kronecker construction.
    Split m=5 into 3+2, p=6 into 3+3.
    Blocks:
      (3x3)*(3x3) -> 3x3 output
      (3x3)*(3x3) -> 3x3 output
      (2x3)*(3x3) -> 2x3 output
      (2x3)*(3x3) -> 2x3 output
    Use schoolbook for each block (simplest correct approach), then see if
    we can do better by exploiting structure. For now, just build the
    block-diagonal schoolbook which is the same as full schoolbook.
    This is a placeholder to test the framework.
    """
    # For a true improvement, we'd need known low-rank decompositions of
    # 3x3x3 (rank 23) and 2x3x3 (rank ~15). The Kronecker product of two
    # decompositions gives a decomposition of the block product.
    # rank(3x3x3) * rank(3x3x3) = 23*23 = 529 for one block pair, which is
    # worse than schoolbook 90. So Kronecker of two 3x3 blocks is not helpful.
    #
    # Instead, let's try a direct structured search for small rank improvements.
    # We'll skip this and go to random search.
    return None


def _random_search(n, m, p, deadline, start_rank=20, end_rank=90):
    """
    Randomized search for a low-rank decomposition.
    For each rank r in [start_rank, end_rank), try several random half-integer
    factor matrices and check if they decompose the tensor.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    best = None
    best_rank = n * m * p  # schoolbook

    for r in range(start_rank, min(end_rank, best_rank)):
        if time.time() > deadline - 10:
            break
        # Number of trials: scale with remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        num_trials = max(20, int(remaining * 3))
        num_trials = min(num_trials, 200)

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            seed = trial * 1000 + r
            U = _random_half_int_matrix(nm, r, seed=seed)
            V = _random_half_int_matrix(mp, r, seed=seed + 1)
            W = _random_half_int_matrix(pn, r, seed=seed + 2)
            if _verify_decomp(n, m, p, U, V, W):
                best = (U, V, W)
                best_rank = r
                break  # found rank r, stop searching this r
        if best_rank <= r:
            break

    return best, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)

    # Try block-Kronecker (currently a no-op placeholder)
    block_result = _try_block_kronecker(n, m, p, deadline)
    if block_result is not None:
        U_b, V_b, W_b = block_result
        if _verify_decomp(n, m, p, U_b, V_b, W_b):
            r_b = U_b.shape[1]
            if r_b < best_rank:
                best_rank = r_b
                best_result = (U_b, V_b, W_b)

    # Random search for lower rank
    rand_result, rand_rank = _random_search(n, m, p, deadline, start_rank=20, end_rank=90)
    if rand_result is not None and rand_rank < best_rank:
        best_rank = rand_rank
        best_result = rand_result

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
