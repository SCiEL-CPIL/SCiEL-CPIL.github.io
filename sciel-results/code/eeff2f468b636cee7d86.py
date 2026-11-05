"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 115.0  # safety margin before 150s kill


def _verify_decomposition(n, m, p, rank, U, V, W):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True


def _apply_strassen_block(n, m, p, U, V, W, col, i0, i1, j0, j1, k0, k1):
    """
    Apply a 2x2x2 Strassen block starting at column 'col'.
    Returns col + 7.
    """
    # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    U[i0 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
    U[i1 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    W[k0 * n + i1][col] = 1
    W[k1 * n + i1][col] = -1
    col += 1

    # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
    U[i0 * m + j0][col] = 1
    V[j0 * p + k1][col] = 1
    V[j1 * p + k1][col] = -1
    W[k1 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
    U[i1 * m + j1][col] = 1
    V[j1 * p + k0][col] = 1
    V[j0 * p + k0][col] = -1
    W[k0 * n + i0][col] = 1
    W[k0 * n + i1][col] = 1
    col += 1

    # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
    U[i0 * m + j0][col] = 1
    U[i0 * m + j1][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = -1
    W[k1 * n + i0][col] = 1
    col += 1

    # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    U[i1 * m + j0][col] = 1
    U[i0 * m + j0][col] = -1
    V[j0 * p + k0][col] = 1
    V[j0 * p + k1][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    U[i0 * m + j1][col] = 1
    U[i1 * m + j1][col] = -1
    V[j1 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    col += 1

    return col


def _build_decomposition(n, m, p, blocks):
    """
    Build a decomposition from a list of Strassen blocks.
    Each block is (i0, i1, j0, j1, k0, k1).
    Uncovered triples get schoolbook terms.
    """
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    uncovered_count = n * m * p - len(covered)
    rank = 7 * len(blocks) + uncovered_count

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block in blocks:
        col = _apply_strassen_block(n, m, p, U, V, W, col, *block)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    if not _verify_decomposition(n, m, p, rank, U, V, W):
        return None

    return {"U": U, "V": V, "W": W}


def _greedy_strassen(n, m, p, t0, max_time=30.0):
    """
    Greedy search: repeatedly pick the block that maximizes new coverage.
    """
    i_pairs = list(itertools.combinations(range(n), 2))
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))

    all_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                all_blocks.append((i0, i1, j0, j1, k0, k1))

    covered = set()
    selected_blocks = []

    while True:
        if time.time() - t0 > _DEADLINE or time.time() - t0 > t0 + max_time:
            break

        best_block = None
        best_gain = 0
        for block in all_blocks:
            i0, i1, j0, j1, k0, k1 = block
            gain = 0
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        if (ii, jj, kk) not in covered:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_gain == 0:
            break

        selected_blocks.append(best_block)
        i0, i1, j0, j1, k0, k1 = best_block
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    result = _build_decomposition(n, m, p, selected_blocks)
    return result


def _local_search_strassen(n, m, p, t0, initial_blocks, max_time=60.0):
    """
    Local search: try to replace blocks or add/remove blocks to reduce rank.
    Rank = 7*len(blocks) + (n*m*p - |covered|)
    """
    def compute_rank(blocks):
        covered = set()
        for (i0, i1, j0, j1, k0, k1) in blocks:
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))
        return 7 * len(blocks) + (n * m * p - len(covered))

    best_blocks = list(initial_blocks)
    best_rank = compute_rank(best_blocks)
    t_start = time.time()

    i_pairs = list(itertools.combinations(range(n), 2))
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))

    all_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                all_blocks.append((i0, i1, j0, j1, k0, k1))

    improved = True
    while improved and (time.time() - t_start) < max_time and (time.time() - t0) < _DEADLINE:
        improved = False

        # Try adding a block
        for new_block in all_blocks:
            if new_block in best_blocks:
                continue
            candidate = best_blocks + [new_block]
            r = compute_rank(candidate)
            if r < best_rank:
                best_blocks = candidate
                best_rank = r
                improved = True

        # Try removing a block
        for idx in range(len(best_blocks)):
            candidate = best_blocks[:idx] + best_blocks[idx+1:]
            r = compute_rank(candidate)
            if r < best_rank:
                best_blocks = candidate
                best_rank = r
                improved = True

        # Try swapping a block
        for idx in range(len(best_blocks)):
            for new_block in all_blocks:
                if new_block == best_blocks[idx]:
                    continue
                candidate = best_blocks[:idx] + [new_block] + best_blocks[idx+1:]
                r = compute_rank(candidate)
                if r < best_rank:
                    best_blocks = candidate
                    best_rank = r
                    improved = True

    result = _build_decomposition(n, m, p, best_blocks)
    return result, best_rank


def _deterministic_strassen(n, m, p):
    """
    Deterministic multi-Strassen decomposition.
    """
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    return _build_decomposition(n, m, p, blocks)


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Start with deterministic baseline
    det_result = _deterministic_strassen(n, m, p)
    if det_result:
        # Extract blocks from deterministic for local search
        initial_blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]
    else:
        initial_blocks = []

    # Try local search to improve
    if (time.time() - t0) < _DEADLINE - 5:
        ls_result, ls_rank = _local_search_strassen(n, m, p, t0, initial_blocks, max_time=50.0)
        if ls_result:
            return ls_result

    # Fallback to deterministic
    if det_result:
        return det_result

    # Last resort: pure schoolbook
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
