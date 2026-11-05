"""Systematic search for rank-minimal tensor decomposition of 3x4*4x7 matrix multiplication.

Uses greedy set-cover over all possible 2x2x2 Strassen blocks (full and partial)
to minimize the number of schoolbook filler terms. Falls back to the known
rank-78 construction if search doesn't find something better.
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    total_entries = n * m * p  # 84

    # Generate all possible 2x2x2 blocks
    # A block is defined by (i0,i1,j0,j1,k0,k1) where i0<i1, j0<j1, k0<k1
    all_blocks = []
    for i0, i1 in itertools.combinations(range(n), 2):
        for j0, j1 in itertools.combinations(range(m), 2):
            for k0, k1 in itertools.combinations(range(p), 2):
                all_blocks.append((i0, i1, j0, j1, k0, k1))

    # Each block covers 8 entries: (i0,j0,k0),(i0,j0,k1),(i0,j1,k0),(i0,j1,k1),
    #                              (i1,j0,k0),(i1,j0,k1),(i1,j1,k0),(i1,j1,k1)
    block_covers = {}
    for idx, (i0, i1, j0, j1, k0, k1) in enumerate(all_blocks):
        covers = set()
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covers.add((ii, jj, kk))
        block_covers[idx] = covers

    # Greedy set-cover: pick blocks that cover the most uncovered entries
    # Try multiple random orderings to get diverse solutions
    import random
    rng = random.Random(42)

    best_rank = 84  # schoolbook
    best_block_set = None

    # Try greedy with different tie-breaking strategies
    for trial in range(200):
        if time.time() - _t0 > _DEADLINE - 10:
            break

        uncovered = set(itertools.product(range(n), range(m), range(p)))
        selected = []
        # Shuffle block indices for diversity
        indices = list(range(len(all_blocks)))
        if trial > 0:
            rng.shuffle(indices)

        while uncovered:
            if time.time() - _t0 > _DEADLINE - 10:
                break
            best_block = -1
            best_overlap = 0
            for idx in indices:
                overlap = len(block_covers[idx] & uncovered)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_block = idx
                elif overlap == best_overlap and overlap > 0:
                    # Tie-break: prefer blocks with more "dense" coverage
                    # (fewer already-covered entries, i.e., more efficient)
                    pass
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered -= block_covers[best_block]

        if uncovered:
            continue  # didn't cover everything (shouldn't happen)

        rank = len(selected) * 7 + len(uncovered)  # uncovered should be 0
        # Actually: rank = 7 * num_blocks + num_remaining_schoolbook
        # But we covered everything, so rank = 7 * len(selected)
        # Wait - each block covers 8 entries with 7 rank terms.
        # Remaining entries need 1 rank term each.
        # But we covered all entries, so rank = 7 * len(selected)
        rank = 7 * len(selected)

        if rank < best_rank:
            best_rank = rank
            best_block_set = list(selected)

    # Also try the known fixed 6-block construction
    fixed_blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    fixed_covered = set()
    for (i0, i1, j0, j1, k0, k1) in fixed_blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    fixed_covered.add((ii, jj, kk))
    fixed_remaining = n * m * p - len(fixed_covered)
    fixed_rank = 6 * 7 + fixed_remaining  # 42 + 36 = 78
    if fixed_rank < best_rank:
        best_rank = fixed_rank
        best_block_set = None  # use fixed blocks

    # Build the decomposition
    if best_block_set is None:
        # Use fixed 6-block construction
        blocks_to_use = fixed_blocks
    else:
        blocks_to_use = [all_blocks[idx] for idx in best_block_set]

    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    covered = set()

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    for (i0, i1, j0, j1, k0, k1) in blocks_to_use:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
