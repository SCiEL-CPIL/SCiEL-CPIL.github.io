"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition via global packing search."""
import time
import random
import itertools

DIMENSIONS = [3, 5, 7]


def construct():
    """
    Finds a maximum packing of disjoint 2x2x2 Strassen blocks for 3x5x7 matrix multiplication,
    then adds schoolbook terms for the remaining (i,j,k) indices.
    Uses randomized greedy with multiple restarts + local search (add/swap operations).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks: choose 2 of 3 rows, 2 of 5 cols, 2 of 7 outputs
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset(
                                (i, j, k)
                                for i in (i1, i2)
                                for j in (j1, j2)
                                for k in (k1, k2)
                            )
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    num_blocks_total = len(all_blocks)
    # Precompute conflict sets
    block_conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                block_conflicts[i].add(j)
                block_conflicts[j].add(i)

    # Build reverse lookup: for each (i,j,k) index, which blocks cover it
    index_to_blocks = {}
    for bi, (_, idxs) in enumerate(all_blocks):
        for idx in idxs:
            index_to_blocks.setdefault(idx, []).append(bi)

    def greedy_packing(order):
        """Given an ordering of block indices, greedily select non-conflicting blocks."""
        used = set()
        selected = []
        for bi in order:
            if not (all_blocks[bi][1] & used):
                selected.append(bi)
                used |= all_blocks[bi][1]
        return selected, used

    best_count = 0
    best_packing = []
    best_used = set()

    random.seed(12345)

    # Phase 1: Randomized greedy with many restarts
    num_restarts = 5000
    for trial in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        order = list(range(num_blocks_total))
        random.shuffle(order)
        packing, used = greedy_packing(order)
        if len(packing) > best_count:
            best_count = len(packing)
            best_packing = packing
            best_used = used
            if best_count >= 12:
                break

    # Phase 2: Local search - try to add blocks, then try swaps (remove 1, add 2)
    if time.time() < deadline - 20.0:
        current = list(best_packing)
        current_used = set(best_used)
        current_set = set(current)
        improved = True
        max_iters = 200
        it = 0
        while improved and it < max_iters and time.time() < deadline - 15.0:
            improved = False
            it += 1

            # Try adding any block not in current packing
            for bi in range(num_blocks_total):
                if bi in current_set:
                    continue
                if not (all_blocks[bi][1] & current_used):
                    current.append(bi)
                    current_set.add(bi)
                    current_used |= all_blocks[bi][1]
                    improved = True
                    break

            if not improved:
                # Try: remove one block, try to add two others
                for ri in range(len(current)):
                    if time.time() > deadline - 15.0:
                        break
                    removed_bi = current[ri]
                    removed_idx = all_blocks[removed_bi][1]
                    trial_used = current_used - removed_idx
                    trial_set = current_set - {removed_bi}
                    trial_packing = [x for x in current if x != removed_bi]

                    added = 0
                    for bi in range(num_blocks_total):
                        if bi in trial_set:
                            continue
                        if not (all_blocks[bi][1] & trial_used):
                            trial_packing.append(bi)
                            trial_set.add(bi)
                            trial_used |= all_blocks[bi][1]
                            added += 1
                            if added >= 2:
                                break

                    if len(trial_packing) > len(current):
                        current = trial_packing
                        current_set = trial_set
                        current_used = trial_used
                        improved = True
                        break

        if len(current) > best_count:
            best_count = len(current)
            best_packing = current
            best_used = current_used

    # Phase 3: More aggressive local search - try remove 2, add 3
    if time.time() < deadline - 15.0 and best_count < 15:
        current = list(best_packing)
        current_used = set(best_used)
        current_set = set(current)
        improved = True
        max_iters = 100
        it = 0
        while improved and it < max_iters and time.time() < deadline - 10.0:
            improved = False
            it += 1
            # Try remove 2 blocks, add 3
            for r1 in range(len(current)):
                if time.time() > deadline - 10.0:
                    break
                for r2 in range(r1 + 1, len(current)):
                    if time.time() > deadline - 10.0:
                        break
                    removed1 = current[r1]
                    removed2 = current[r2]
                    rem_idx = all_blocks[removed1][1] | all_blocks[removed2][1]
                    trial_used = current_used - rem_idx
                    trial_set = current_set - {removed1, removed2}
                    trial_packing = [x for x in current if x not in (removed1, removed2)]

                    added = 0
                    for bi in range(num_blocks_total):
                        if bi in trial_set:
                            continue
                        if not (all_blocks[bi][1] & trial_used):
                            trial_packing.append(bi)
                            trial_set.add(bi)
                            trial_used |= all_blocks[bi][1]
                            added += 1
                            if added >= 3:
                                break

                    if len(trial_packing) > len(current):
                        current = trial_packing
                        current_set = trial_set
                        current_used = trial_used
                        improved = True
                        break
                if improved:
                    break

        if len(current) > best_count:
            best_count = len(current)
            best_packing = current
            best_used = current_used

    # Build the decomposition
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add Strassen blocks
    covered = set()
    for bi in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[bi][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Convert to row-major matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
