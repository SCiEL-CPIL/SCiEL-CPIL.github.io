"""Optimized rank decomposition for 3x4*4x7 matrix multiplication.

Strategy:
- Exhaustive search over all maximal packings of disjoint 2x2x2 Strassen blocks
- Each block reduces rank by 1 (from 8 schoolbook to 7 Strassen)
- Maximize number of blocks to minimize total rank
- rank = 84 - num_blocks (since each block saves 1 multiplication)

For 3x4x7: 36 possible blocks, exhaustive backtracking with pruning
"""
import time
import sys

_DIMS = (3, 4, 7)
_DEADLINE = 140.0  # Safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = _DIMS

    # All possible 2x2x2 blocks
    all_blocks = [(i0, j0, k0) for i0 in range(n - 1)
                  for j0 in range(m - 1)
                  for k0 in range(p - 1)]
    num_blocks_total = len(all_blocks)  # 2*3*6 = 36

    # Entry indexing: (i,j,k) -> 0..83
    entry_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_to_idx[(i, j, k)] = idx
                idx += 1
    total_entries = n * m * p  # 84

    # Bitmask for each block (8 entries)
    block_masks = []
    for (i0, j0, k0) in all_blocks:
        mask = 0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    mask |= (1 << entry_to_idx[(i0 + di, j0 + dj, k0 + dk)])
        block_masks.append(mask)

    # Precompute disjointness matrix
    disjoint = [[True] * num_blocks_total for _ in range(num_blocks_total)]
    for a in range(num_blocks_total):
        for b in range(a + 1, num_blocks_total):
            disjoint[a][b] = (block_masks[a] & block_masks[b]) == 0
            disjoint[b][a] = disjoint[a][b]

    # Full mask
    full_mask = (1 << total_entries) - 1

    best_packing = []
    best_size = 0

    # Order blocks by some heuristic: blocks that conflict with fewer others first
    # Actually for maximum independent set, we want to try blocks that are "rare" first
    # Sort by number of conflicts (ascending) - blocks with fewer conflicts are more flexible
    conflict_count = [sum(1 for b in range(num_blocks_total) if a != b and not disjoint[a][b])
                      for a in range(num_blocks_total)]
    block_order = sorted(range(num_blocks_total), key=lambda x: conflict_count[x])

    # Map from position in order to block index
    order_to_block = block_order
    block_to_order = [0] * num_blocks_total
    for pos, blk in enumerate(block_order):
        block_to_order[blk] = pos

    def backtrack(start_pos, current, current_mask, current_set):
        nonlocal best_packing, best_size
        if time.time() - _t0 > _DEADLINE:
            return

        if len(current) > best_size:
            best_size = len(current)
            best_packing = current[:]

        # Upper bound: remaining entries / 8
        remaining = full_mask & ~current_mask
        remaining_count = bin(remaining).count('1')
        if len(current) + remaining_count // 8 <= best_size:
            return

        for pos in range(start_pos, num_blocks_total):
            if time.time() - _t0 > _DEADLINE:
                return
            blk = order_to_block[pos]
            if current_mask & block_masks[blk]:
                continue
            current.append(blk)
            current_set.add(blk)
            backtrack(pos + 1, current, current_mask | block_masks[blk], current_set)
            current.pop()
            current_set.discard(blk)

    backtrack(0, [], 0, set())

    # Build decomposition
    num_strassen = len(best_packing)
    rank = total_entries - num_strassen  # 84 - num_blocks

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0 + 1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0 * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for bi in best_packing:
        add_strassen(*all_blocks[bi])

    # Add schoolbook terms for uncovered entries
    covered = set()
    for bi in best_packing:
        i0, j0, k0 = all_blocks[bi]
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    assert col == rank, f"Rank mismatch: {col} vs {rank}"

    # Quick verification with numpy if time permits
    if time.time() - _t0 < _DEADLINE - 10.0:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = float(np.dot(U_np[u_idx] * V_np[v_idx], W_np[w_idx]))
                    if abs(val - 1.0) > 1e-9:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): {val}")

    return {"U": U, "V": V, "W": W}
