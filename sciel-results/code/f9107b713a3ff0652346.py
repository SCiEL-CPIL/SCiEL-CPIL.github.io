import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses branch-and-bound maximum independent set with bitset optimization to find
    the maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | frozenset([(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    num_blocks_total = len(all_blocks)

    # Build conflict graph using bitmasks
    # Two blocks conflict if they share any (i,j,k) index
    # Precompute bitmask for each block
    all_indices = set()
    for _, indices in all_blocks:
        all_indices |= indices
    index_to_bit = {idx: bit for bit, idx in enumerate(all_indices)}
    num_bits = len(all_indices)

    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << index_to_bit[idx])
        block_bitmasks.append(mask)

    # Build adjacency: conflicts[i] is bitmask of blocks that conflict with block i
    conflicts_mask = [0] * num_blocks_total
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflicts_mask[i] |= (1 << j)
                conflicts_mask[j] |= (1 << i)

    # Degrees
    degrees = [bin(conflicts_mask[i]).count('1') for i in range(num_blocks_total)]

    # Sort blocks by degree ascending (fewer conflicts first) for better ordering
    sorted_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])

    # Branch and bound for maximum independent set
    # We work with the original block indices
    best_set = []
    best_size = 0

    def branch_and_bound(available_mask, current_size, current_set):
        nonlocal best_set, best_size

        if time.time() > deadline - 5.0:
            return

        # Upper bound: current_size + popcount(available_mask)
        upper_bound = current_size + bin(available_mask).count('1')
        if upper_bound <= best_size:
            return

        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return

        # Pick a vertex with minimum degree in the available subgraph
        # Try each available vertex
        avail_list = []
        temp_mask = available_mask
        while temp_mask:
            # Extract lowest set bit
            lsb = temp_mask & (-temp_mask)
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            temp_mask ^= lsb

        # Order by degree in available subgraph (ascending)
        def avail_degree(idx):
            # Count conflicts of idx within available_mask
            return bin(conflicts_mask[idx] & available_mask).count('1')

        avail_list.sort(key=avail_degree)

        for idx in avail_list:
            if time.time() > deadline - 5.0:
                return

            if not (available_mask & (1 << idx)):
                continue

            # Branch 1: Include idx
            new_available = available_mask & ~(1 << idx) & ~conflicts_mask[idx]
            branch_and_bound(new_available, current_size + 1, current_set + [idx])

            # Branch 2: Exclude idx
            available_mask = available_mask & ~(1 << idx)

            # Pruning: if we can't beat best even with all remaining
            if current_size + bin(available_mask).count('1') <= best_size:
                break

    # Run branch and bound
    initial_available = (1 << num_blocks_total) - 1
    branch_and_bound(initial_available, 0, [])

    # If B&B didn't find good solution, try greedy with multiple restarts
    if best_size < 12:
        def greedy_packing(order):
            used_mask = 0
            current = []
            for idx in order:
                if not (block_bitmasks[idx] & used_mask):
                    current.append(idx)
                    used_mask |= block_bitmasks[idx]
            return current

        def local_improve(packing):
            if not packing:
                return packing

            best_packing = packing[:]
            best_mask = 0
            for idx in best_packing:
                best_mask |= block_bitmasks[idx]

            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 3.0:
                    break

                # Try removing 1 block and adding 2
                for pos in range(len(best_packing)):
                    if time.time() > deadline - 3.0:
                        break
                    remaining = best_packing[:pos] + best_packing[pos+1:]
                    used_remaining_mask = 0
                    for idx in remaining:
                        used_remaining_mask |= block_bitmasks[idx]

                    # Find blocks compatible with remaining
                    candidates = []
                    for idx in range(num_blocks_total):
                        if idx not in remaining and not (block_bitmasks[idx] & used_remaining_mask):
                            candidates.append(idx)

                    # Try adding 2 compatible candidates
                    found_pair = None
                    for i in range(len(candidates)):
                        c1 = candidates[i]
                        mask_c1 = used_remaining_mask | block_bitmasks[c1]
                        for j in range(i + 1, len(candidates)):
                            c2 = candidates[j]
                            if not (block_bitmasks[c2] & mask_c1):
                                found_pair = (c1, c2)
                                break
                        if found_pair:
                            break

                    if found_pair:
                        new_packing = remaining + list(found_pair)
                        if len(new_packing) > len(best_packing):
                            best_packing = new_packing
                            best_mask = 0
                            for idx in best_packing:
                                best_mask |= block_bitmasks[idx]
                            improved = True
                            break

                # Try removing 2 blocks and adding 3
                if not improved and len(best_packing) >= 2:
                    for pos1 in range(len(best_packing)):
                        if time.time() > deadline - 3.0:
                            break
                        for pos2 in range(pos1 + 1, len(best_packing)):
                            remaining = [x for i, x in enumerate(best_packing) if i != pos1 and i != pos2]
                            used_remaining_mask = 0
                            for idx in remaining:
                                used_remaining_mask |= block_bitmasks[idx]

                            candidates = []
                            for idx in range(num_blocks_total):
                                if idx not in remaining and not (block_bitmasks[idx] & used_remaining_mask):
                                    candidates.append(idx)

                            found_triple = None
                            for i in range(len(candidates)):
                                c1 = candidates[i]
                                mask_c1 = used_remaining_mask | block_bitmasks[c1]
                                for j in range(i + 1, len(candidates)):
                                    c2 = candidates[j]
                                    if block_bitmasks[c2] & mask_c1:
                                        continue
                                    mask_c2 = mask_c1 | block_bitmasks[c2]
                                    for k in range(j + 1, len(candidates)):
                                        c3 = candidates[k]
                                        if not (block_bitmasks[c3] & mask_c2):
                                            found_triple = (c1, c2, c3)
                                            break
                                    if found_triple:
                                        break
                                if found_triple:
                                    break

                            if found_triple:
                                new_packing = remaining + list(found_triple)
                                if len(new_packing) > len(best_packing):
                                    best_packing = new_packing
                                    best_mask = 0
                                    for idx in best_packing:
                                        best_mask |= block_bitmasks[idx]
                                    improved = True
                                    break
                        if improved:
                            break

            return best_packing

        # Multiple greedy restarts
        num_trials = 2000
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                break

            # Different ordering strategies
            if trial % 4 == 0:
                order = sorted_order[:]
            elif trial % 4 == 1:
                order = sorted(range(num_blocks_total), key=lambda x: -degrees[x])
            elif trial % 4 == 2:
                order = sorted_order[:]
                random.shuffle(order)
            else:
                order = sorted_order[:]
                # Perturb
                for _ in range(random.randint(1, 5)):
                    i = random.randint(0, len(order) - 1)
                    j = random.randint(0, len(order) - 1)
                    order[i], order[j] = order[j], order[i]

            packing = greedy_packing(order)

            if len(packing) > best_size:
                best_size = len(packing)
                best_set = packing[:]

                if time.time() < deadline - 3.0:
                    improved = local_improve(best_set)
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_set = improved

                if best_size >= 13:
                    break

    # Construct U, V, W based on the best packing found
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

    # Add the Strassen blocks from the best packing
    covered = set()
    for idx in best_set:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Convert to matrix format
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
# EVOLVE-BLOCK-END
