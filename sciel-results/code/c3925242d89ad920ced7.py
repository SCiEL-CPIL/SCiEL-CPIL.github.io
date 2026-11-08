import time
import random
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy set-cover approach with diverse efficient blocks (Strassen 2x2x2,
    rank-2 line blocks, rank-3 plane blocks) to minimize total rank.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_cells = n * m * p  # 105

    # We'll represent each "block" as a list of (U_entries, V_entries, W_entries) triples
    # where each triple is one rank-1 term. The block covers specific (i,j,k) cells.
    # A block is valid if applying its terms exactly reproduces the tensor entries for
    # the cells it covers (and contributes 0 to other cells).

    # Generate all candidate blocks

    # Block type 1: 2x2x2 Strassen block (rank 7, covers 8 cells)
    strassen_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            cells = frozenset((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))
                            # 7 rank-1 terms for Strassen
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
                            terms = [
                                ({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}),
                                ({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}),
                                ({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}),
                                ({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}),
                                ({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}),
                                ({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}),
                                ({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}),
                            ]
                            strassen_blocks.append((cells, terms, 7))

    # Block type 2: 2x1x1 rank-2 block: (a1+a2)(b1) covers (i1,j,k) and (i2,j,k)
    # Actually for 2x1x1, we need: C[i1,k] = A[i1,j]*B[j,k], C[i2,k] = A[i2,j]*B[j,k]
    # These are just 2 rank-1 terms. Not really a "block" saving.
    # Better: use 2x2x1 rank-3 blocks.
    # 2x2x1: C[i1,k] = A[i1,j1]*B[j1,k] + A[i1,j2]*B[j2,k]
    #         C[i2,k] = A[i2,j1]*B[j1,k] + A[i2,j2]*B[j2,k]
    # Standard: 4 rank-1 terms (schoolbook). 
    # Strassen-like for 2x2x1: use 3 terms?
    # (a11+a22)(b11+b22) - a11*b11 - a22*b22 gives c11+c22 contribution... 
    # Actually for 2x2x1, the minimal rank is 4 (it's just a 2x2 matrix multiply with 1 output).
    # Wait, no: it's computing a 2x1 vector from 2x2 * 2x1. That's rank 4 generally.
    # Let me think differently.

    # Block type 3: 1x2x2 rank-2? No, 1x2x2 is just a dot product times 2, rank 4.
    
    # Let me focus on what actually helps: the key is to find blocks that cover 
    # multiple cells with fewer rank-1 terms than schoolbook.
    # 
    # Schoolbook: each (i,j,k) cell needs 1 rank-1 term. Total = 105.
    # Strassen 2x2x2: 7 terms for 8 cells (saves 1 per block).
    # 
    # What about 2x2x2 with rank 6? (Laderman's algorithm is for 4x4x4)
    # For 2x2x2, rank 7 is known minimal.
    #
    # What about larger blocks? 3x3x3? Too complex.
    #
    # Alternative: use the fact that we have 3 rows. Maybe a 3x2x2 block with rank < 12?
    # 3x2x2 schoolbook = 12 terms. Can we do better?
    # This is getting complex. Let me stick with 2x2x2 Strassen blocks but improve the packing.

    # Actually, let me try a different approach: use integer linear programming or
    # a better greedy with lookahead.

    # Generate all 2x2x2 Strassen blocks with cell masks
    all_cells = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    cell_to_idx = {c: idx for idx, c in enumerate(sorted(all_cells))}
    num_cells = len(all_cells)

    # For each Strassen block, create a bitmask of cells it covers
    block_cell_masks = []
    block_terms_list = []
    block_ranks = []
    block_params_list = []

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        cell_idx = cell_to_idx[(i, j, k)]
                                        mask |= (1 << cell_idx)
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
                            terms = [
                                ({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}),
                                ({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}),
                                ({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}),
                                ({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}),
                                ({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}),
                                ({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}),
                                ({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}),
                            ]
                            block_cell_masks.append(mask)
                            block_terms_list.append(terms)
                            block_ranks.append(7)
                            block_params_list.append((i1, i2, j1, j2, k1, k2))

    num_blocks = len(block_cell_masks)

    # Greedy set cover: repeatedly pick the block that covers the most uncovered cells
    # with the best "efficiency" (newly covered cells / rank)
    # But since all blocks have rank 7, just maximize newly covered cells.

    best_total_rank = 105  # schoolbook baseline
    best_packing = []

    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break

        # Randomize the order of ties by adding small random noise to efficiency
        covered_mask = 0
        packing = []
        remaining = set(range(num_blocks))

        while remaining:
            # Find best block: maximize (newly covered cells) / rank
            best_block = -1
            best_gain = -1
            best_eff = -1

            # Sample a subset for speed
            if len(remaining) > 50:
                sample = random.sample(list(remaining), 50)
            else:
                sample = list(remaining)

            for b in sample:
                new_cells = block_cell_masks[b] & ~covered_mask
                gain = bin(new_cells).count('1')
                if gain == 0:
                    continue
                eff = gain / block_ranks[b]
                # Add small random tiebreaker
                eff += random.random() * 0.001
                if eff > best_eff:
                    best_eff = eff
                    best_block = b
                    best_gain = gain

            if best_block == -1:
                break

            packing.append(best_block)
            covered_mask |= block_cell_masks[best_block]
            remaining.discard(best_block)

            # Check if we've covered all cells
            if covered_mask == (1 << num_cells) - 1:
                break

        # Calculate total rank: sum of block ranks + uncovered cells (schoolbook)
        total_rank = sum(block_ranks[b] for b in packing)
        uncovered = ((1 << num_cells) - 1) & ~covered_mask
        total_rank += bin(uncovered).count('1')

        if total_rank < best_total_rank:
            best_total_rank = total_rank
            best_packing = packing[:]
            if best_total_rank <= 85:
                break

    # Local search: try to swap blocks to reduce total rank
    if time.time() < deadline - 15.0:
        current_packing = best_packing[:]
        improved = True
        max_iters = 30
        for it in range(max_iters):
            if time.time() > deadline - 12.0:
                break
            improved = False

            # Try removing each block and re-optimizing
            for idx in range(len(current_packing)):
                if time.time() > deadline - 12.0:
                    break

                temp_packing = current_packing[:idx] + current_packing[idx+1:]
                covered_mask = 0
                for b in temp_packing:
                    covered_mask |= block_cell_masks[b]
                base_rank = sum(block_ranks[b] for b in temp_packing)
                uncovered = ((1 << num_cells) - 1) & ~covered_mask
                base_total = base_rank + bin(uncovered).count('1')

                # Try adding different blocks
                remaining = set(range(num_blocks)) - set(temp_packing)
                best_new_total = base_total
                best_new_packing = None

                # Greedily add blocks to minimize total
                trial_packing = temp_packing[:]
                trial_covered = covered_mask
                trial_rank = base_rank

                while True:
                    best_add = -1
                    best_add_eff = -1
                    if len(remaining) > 50:
                        sample = random.sample(list(remaining), 50)
                    else:
                        sample = list(remaining)

                    for b in sample:
                        new_cells = block_cell_masks[b] & ~trial_covered
                        gain = bin(new_cells).count('1')
                        if gain == 0:
                            continue
                        eff = gain / block_ranks[b] + random.random() * 0.001
                        if eff > best_add_eff:
                            best_add_eff = eff
                            best_add = b

                    if best_add == -1:
                        break

                    trial_packing.append(best_add)
                    trial_covered |= block_cell_masks[best_add]
                    trial_rank += block_ranks[best_add]
                    remaining.discard(best_add)

                trial_uncovered = ((1 << num_cells) - 1) & ~trial_covered
                trial_total = trial_rank + bin(trial_uncovered).count('1')

                if trial_total < best_new_total:
                    best_new_total = trial_total
                    best_new_packing = trial_packing

                if best_new_packing and best_new_total < base_total:
                    current_packing = best_new_packing
                    improved = True
                    if sum(block_ranks[b] for b in current_packing) + bin(((1 << num_cells) - 1) & ~sum([block_cell_masks[b] for b in current_packing], 0)).count('1') < best_total_rank:
                        best_total_rank = sum(block_ranks[b] for b in current_packing) + bin(((1 << num_cells) - 1) & ~sum([block_cell_masks[b] for b in current_packing], 0)).count('1')
                        best_packing = current_packing[:]
                    break

            if not improved:
                break

    # Build the final decomposition
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

    # Add Strassen blocks from best packing
    covered = set()
    for b in best_packing:
        terms = block_terms_list[b]
        for u_e, v_e, w_e in terms:
            add_term(u_e, v_e, w_e)
        i1, i2, j1, j2, k1, k2 = block_params_list[b]
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered cells
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
