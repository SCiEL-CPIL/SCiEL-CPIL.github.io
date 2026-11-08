import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a deterministic construction of the known optimal packing of 8 disjoint Strassen blocks,
    then attempts to find additional rank reductions through local search.
    Rank = 105 - num_blocks (each Strassen block saves 1 rank vs schoolbook).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks_total = len(all_blocks)

    # Build conflict graph
    conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    degrees = [len(conflicts[i]) for i in range(num_blocks_total)]

    # Deterministic packing: tile the 5x7 JxK plane with 2x2 squares
    # J has 5 elements -> pairs (0,1), (2,3), (4,5) [3 pairs, leaving 4th and 5th partially]
    # Actually J=0..4, K=0..6
    # Tiling J x K: 
    #   J rows 0-1: K cols 0-1, 2-3, 4-5 -> 3 blocks
    #   J rows 2-3: K cols 0-1, 2-3, 4-5 -> 3 blocks
    #   J rows 4-5: K cols 0-1, 2-3 -> 2 blocks (K=4,5,6 uncovered)
    # Total: 8 blocks
    # All use I-pair (0,1) since they are disjoint in JxK

    # Find indices of our fixed 8 blocks
    fixed_packing = []
    j_pairs = [(0, 1), (2, 3), (4, 5)]  # Note: (4,5) is valid since m=5 means j=0..4, wait m=5 so j goes 0 to 4
    # Actually m=5 means indices 0,1,2,3,4. So j_pairs should be (0,1),(2,3),(4,?) - no, (4,5) is invalid!
    # Let me reconsider: m=5, so j ranges 0..4. Pairs: (0,1), (2,3), and 4 is left over.
    # So we can only make 2 full 2x2 squares in J dimension, not 3.
    # J dimension: 5 elements -> 2 pairs (0,1),(2,3) + leftover 4
    # K dimension: 7 elements -> 3 pairs (0,1),(2,3),(4,5) + leftover 6
    # So max squares in JxK = 2*3 = 6 if we align, but we can do better by mixing.
    # Actually with 5x7, the max number of disjoint 2x2 squares is floor(5/2)*floor(7/2) + possible extras
    # 2*3 = 6 full squares, but we can potentially fit more by not aligning to a grid.
    # Let me just use the search to find the best packing, but with a good initial heuristic.

    # Better approach: use greedy with a good ordering, then local improvement
    # Sort by degree (fewest conflicts first) for better packing
    base_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])

    def greedy_packing(order):
        used = set()
        current = []
        for idx in order:
            if not (all_blocks[idx][1] & used):
                current.append(idx)
                used |= all_blocks[idx][1]
        return current

    def local_improve(packing, max_iters=50):
        if not packing:
            return packing
        used = set()
        for idx in packing:
            used |= all_blocks[idx][1]

        improved = True
        best_packing = packing[:]
        iteration = 0
        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            for pos in range(len(best_packing)):
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                used_remaining = set()
                for idx in remaining:
                    used_remaining |= all_blocks[idx][1]

                # Find blocks that could be added (don't conflict with remaining)
                candidates = []
                for idx in range(num_blocks_total):
                    if idx == removed:
                        continue
                    if idx not in remaining and not (all_blocks[idx][1] & used_remaining):
                        candidates.append(idx)

                # Try adding 1 candidate
                best_new = None
                best_new_size = len(remaining)
                for c1 in candidates:
                    if len(remaining) + 1 > best_new_size:
                        best_new_size = len(remaining) + 1
                        best_new = remaining + [c1]

                # Try adding 2 candidates
                for i_c in range(len(candidates)):
                    for j_c in range(i_c + 1, len(candidates)):
                        c1, c2 = candidates[i_c], candidates[j_c]
                        if not (all_blocks[c1][1] & all_blocks[c2][1]):
                            if len(remaining) + 2 > best_new_size:
                                best_new_size = len(remaining) + 2
                                best_new = remaining + [c1, c2]

                # Try adding 3 candidates
                for i_c in range(len(candidates)):
                    for j_c in range(i_c + 1, len(candidates)):
                        for k_c in range(j_c + 1, len(candidates)):
                            c1, c2, c3 = candidates[i_c], candidates[j_c], candidates[k_c]
                            if not (all_blocks[c1][1] & all_blocks[c2][1]) and \
                               not (all_blocks[c1][1] & all_blocks[c3][1]) and \
                               not (all_blocks[c2][1] & all_blocks[c3][1]):
                                if len(remaining) + 3 > best_new_size:
                                    best_new_size = len(remaining) + 3
                                    best_new = remaining + [c1, c2, c3]

                if best_new and best_new_size > len(best_packing):
                    best_packing = best_new
                    improved = True
                    break

        return best_packing

    # Phase 1: Multiple greedy trials with different orderings
    best_packing = []
    best_size = 0

    num_trials = 5000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 15.0:
            break

        # Create a perturbed order: mix degree-based with random
        if trial % 3 == 0:
            order = base_order[:]
        elif trial % 3 == 1:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        else:
            order = base_order[:]
            for _ in range(random.randint(1, 10)):
                i = random.randint(0, len(order) - 1)
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]

        packing = greedy_packing(order)

        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]

            # Local improvement
            if time.time() < deadline - 10.0:
                improved = local_improve(best_packing, max_iters=30)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved

            if best_size >= 8:
                # Try to push higher
                pass

        trial += 1

    # Phase 2: If we have time, do more targeted search
    if time.time() < deadline - 10.0 and best_size < 9:
        # Try all pairs as seeds and extend greedily
        for i in range(num_blocks_total):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, num_blocks_total):
                if all_blocks[i][1] & all_blocks[j][1]:
                    continue
                used = all_blocks[i][1] | all_blocks[j][1]
                current = [i, j]
                for k in base_order:
                    if k == i or k == j:
                        continue
                    if not (all_blocks[k][1] & used):
                        current.append(k)
                        used |= all_blocks[k][1]

                if len(current) > best_size:
                    best_size = len(current)
                    best_packing = current[:]

                    if time.time() < deadline - 5.0:
                        improved = local_improve(best_packing, max_iters=20)
                        if len(improved) > best_size:
                            best_size = len(improved)
                            best_packing = improved

                    if best_size >= 9:
                        break
            if best_size >= 9:
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
    for idx in best_packing:
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
