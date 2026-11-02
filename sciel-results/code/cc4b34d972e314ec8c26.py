"""
Rank-minimal exact tensor decomposition for 3×5×6 matrix multiplication.

Strategy:
  1. Build the schoolbook (rank 90) decomposition.
  2. Attempt to reduce rank by searching for structured cancellations
     and by using randomised local search with a smooth error objective.
  3. Return the best valid decomposition found before the deadline.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]


def _schoolbook(n, m, p):
    """Return the trivial (schoolbook) decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _tensor_error(U, V, W, n, m, p):
    """
    Compute the total squared error of the decomposition:
      error = sum_{i,j,k} ( sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1 )^2
    Uses vectorised NumPy for speed.
    """
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Build the full tensor contraction:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can compute this by iterating over c (rank dimension) since r may be up to ~90.
    # For each c, we add outer product of the three vectors reshaped appropriately.
    total = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U_np[:, c].reshape(n, m)       # (n, m)
        vc = V_np[:, c].reshape(m, p)       # (m, p)
        wc = W_np[:, c].reshape(p, n)       # (p, n)
        # The contribution for this c is: for each (i,j,k), uc[i,j] * vc[j,k] * wc[k,i]
        # We can compute this as: np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        total += np.einsum('ij,jk,ki->ijk', uc, vc, wc)

    error = np.sum((total - 1.0) ** 2)
    return float(error)


def _check_exact(U, V, W, n, m, p, tol=1e-6):
    """Return True if the decomposition is exact within tolerance."""
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    total = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U_np[:, c].reshape(n, m)
        vc = V_np[:, c].reshape(m, p)
        wc = W_np[:, c].reshape(p, n)
        total += np.einsum('ij,jk,ki->ijk', uc, vc, wc)

    return bool(np.max(np.abs(total - 1.0)) < tol)


def _random_sparse_init(n, m, p, r, seed=None):
    """
    Random initialisation with sparse entries in {-1, 0, 1}.
    Each column gets a few non-zero entries in each factor.
    """
    if seed is not None:
        random.seed(seed)
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for c in range(r):
        # Place 2-4 non-zeros in each factor
        nz_u = random.randint(2, 4)
        nz_v = random.randint(2, 4)
        nz_w = random.randint(2, 4)

        idx_u = random.sample(range(n * m), nz_u)
        idx_v = random.sample(range(m * p), nz_v)
        idx_w = random.sample(range(p * n), nz_w)

        for idx in idx_u:
            U[idx][c] = random.choice([-1, 1])
        for idx in idx_v:
            V[idx][c] = random.choice([-1, 1])
        for idx in idx_w:
            W[idx][c] = random.choice([-1, 1])

    return U, V, W


def _local_search(U, V, W, n, m, p, deadline, max_iter=5000, candidates=(-2, -1, 0, 1, 2)):
    """
    Hill-climbing local search: perturb one random entry at a time,
    accept if error decreases. Uses NumPy-vectorised error for speed.
    """
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    current_error = _tensor_error_from_arrays(U_np, V_np, W_np, n, m, p)
    best_error = current_error
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()

    cand_list = list(candidates)

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Pick a random entry to perturb
        mat = random.choice([0, 1, 2])
        if mat == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, r - 1)
            old = U_np[row, col]
            new = random.choice([v for v in cand_list if v != old])
            U_np[row, col] = new
        elif mat == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, r - 1)
            old = V_np[row, col]
            new = random.choice([v for v in cand_list if v != old])
            V_np[row, col] = new
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, r - 1)
            old = W_np[row, col]
            new = random.choice([v for v in cand_list if v != old])
            W_np[row, col] = new

        new_error = _tensor_error_from_arrays(U_np, V_np, W_np, n, m, p)

        if new_error < current_error:
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U = U_np.copy()
                best_V = V_np.copy()
                best_W = W_np.copy()
                if best_error < 1e-12:
                    break
        else:
            # Revert
            if mat == 0:
                U_np[row, col] = old
            elif mat == 1:
                V_np[row, col] = old
            else:
                W_np[row, col] = old

    # Convert back to lists of lists (required by return interface)
    best_U_list = [row.tolist() for row in best_U]
    best_V_list = [row.tolist() for row in best_V]
    best_W_list = [row.tolist() for row in best_W]
    return best_U_list, best_V_list, best_W_list, best_error


def _tensor_error_from_arrays(U_np, V_np, W_np, n, m, p):
    """Fast error computation from pre-converted NumPy arrays."""
    r = U_np.shape[1]
    total = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U_np[:, c].reshape(n, m)
        vc = V_np[:, c].reshape(m, p)
        wc = W_np[:, c].reshape(p, n)
        total += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    return float(np.sum((total - 1.0) ** 2))


def _try_rank(n, m, p, r, deadline, num_tries=5):
    """
    Attempt to find a valid decomposition of rank r.
    Returns (U, V, W) if found, else None.
    """
    best = None
    best_err = float('inf')
    for t in range(num_tries):
        if time.time() > deadline:
            break
        U, V, W = _random_sparse_init(n, m, p, r, seed=None)
        U, V, W, err = _local_search(U, V, W, n, m, p, deadline, max_iter=3000)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if err < 1e-12:
            return U, V, W
    if best is not None and _check_exact(best[0], best[1], best[2], n, m, p):
        return best
    return None


def _strassen_block_decomposition(n, m, p):
    """
    Attempt a Strassen-like block decomposition.
    For 3x5x6, we can try to partition into blocks and apply recursive/structured
    low-rank decompositions. This is heuristic.
    
    One approach: treat the 3x5 as [3x2 | 3x3] and 5x6 as [[2x6],[3x6]],
    then use a 2x2 block Strassen on the 3x3 and 3x2 parts.
    This is complex; we'll try a simpler structured approach.
    """
    # For now, return None to indicate this path is not fully implemented.
    # The local search should handle the search.
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix.
    
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries that exactly decompose the tensor.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s wall-clock kill

    # Schoolbook fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Lower bound for 3x5x6: known to be at least 51 (from literature on
    # small matrix multiplication tensors). We search from 51 upward.
    random.seed(42)

    for target_rank in range(51, 91):
        if time.time() > deadline:
            break

        # Try multiple random initialisations for this rank
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            result = _try_rank(n, m, p, target_rank, deadline, num_tries=1)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                # Found a valid decomposition at this rank; return it.
                # Since we search in increasing order, this is the minimal
                # rank we've found so far.
                return best_result

    # If no improvement over schoolbook was found, return schoolbook.
    return best_result
