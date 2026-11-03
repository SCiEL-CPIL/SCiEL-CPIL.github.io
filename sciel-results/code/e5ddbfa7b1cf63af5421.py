import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21

    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    def strassen_factors(i0, i1, j0, j1, k0, k1):
        """
        Generate the 7 Strassen rank-1 terms for a 2x2x2 block.
        Returns list of (U_dict, V_dict, W_dict) where each dict maps
        flattened index -> coefficient.
        """
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1

        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1

        c00 = k0 * n + i0
        c01 = k0 * n + i1
        c10 = k1 * n + i0
        c11 = k1 * n + i1

        terms = []

        # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1]) -> C00 += M1, C11 += M1
        terms.append(({a00: 1.0, a11: 1.0}, {b00: 1.0, b11: 1.0}, {c00: 1.0, c11: 1.0}))

        # M2: (A[i1,j0]+A[i1,j1])B[j0,k0] -> C10 += M2, C11 -= M2
        terms.append(({a10: 1.0, a11: 1.0}, {b00: 1.0}, {c10: 1.0, c11: -1.0}))

        # M3: A[i0,j0](B[j0,k1]-B[j1,k1]) -> C01 += M3, C11 += M3
        terms.append(({a00: 1.0}, {b01: 1.0, b11: -1.0}, {c01: 1.0, c11: 1.0}))

        # M4: A[i1,j1](B[j1,k0]-B[j0,k0]) -> C00 += M4, C10 += M4
        terms.append(({a11: 1.0}, {b10: 1.0, b00: -1.0}, {c00: 1.0, c10: 1.0}))

        # M5: (A[i0,j0]+A[i0,j1])B[j1,k1] -> C00 -= M5, C01 += M5
        terms.append(({a00: 1.0, a01: 1.0}, {b11: 1.0}, {c00: -1.0, c01: 1.0}))

        # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1]) -> C11 += M6
        terms.append(({a10: 1.0, a00: -1.0}, {b00: 1.0, b01: 1.0}, {c11: 1.0}))

        # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1]) -> C00 += M7
        terms.append(({a01: 1.0, a11: -1.0}, {b10: 1.0, b11: 1.0}, {c00: 1.0}))

        return terms

    # Generate all possible 2x2x2 blocks (consecutive indices)
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1))
                all_blocks.append((i0, i1, j0, j1, k0, k1, entries))

    num_blocks = len(all_blocks)

    # Build conflict graph using bitmasks
    conflict_mask = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][6] & all_blocks[j][6]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def get_bit_count(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    # Backtracking to find maximum independent set
    best_set = []
    best_size = 0
    time_check_counter = [0]

    def backtrack(available_mask, current_set):
        nonlocal best_set, best_size
        time_check_counter[0] += 1
        if time_check_counter[0] % 50 == 0 and time.time() > deadline - 30:
            return

        avail_count = get_bit_count(available_mask)
        if len(current_set) + avail_count <= best_size:
            return

        if available_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return

        # Choose the most constrained available block
        max_conf = -1
        best_idx = -1
        mask = available_mask
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            mask ^= lsb
            conf_in_avail = get_bit_count(conflict_mask[idx] & available_mask) - 1
            if conf_in_avail > max_conf:
                max_conf = conf_in_avail
                best_idx = idx

        if best_idx < 0:
            return

        remaining_mask = available_mask & ~(1 << best_idx)

        # Option 1: include block best_idx
        current_set.append(best_idx)
        new_available = remaining_mask & ~conflict_mask[best_idx]
        backtrack(new_available, current_set)
        current_set.pop()

        # Option 2: exclude block best_idx
        backtrack(remaining_mask, current_set)

    backtrack((1 << num_blocks) - 1, [])

    selected_indices = list(best_set)

    # Local search: try to improve the set via swaps and additions
    def try_local_search(selected, deadline):
        """Try to improve the selected set via local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            selected_set = set(selected)
            covered = set()
            for idx in selected:
                covered |= all_blocks[idx][6]

            # Try adding any block that doesn't conflict
            for idx in range(num_blocks):
                if idx in selected_set:
                    continue
                if all_blocks[idx][6] & covered:
                    continue
                # Adding this block: check if we can drop a conflicting block
                # For now, just try pure addition (shouldn't happen if max)
                # Try swap: drop one selected, add this one
                for drop_idx in selected:
                    if drop_idx == idx:
                        continue
                    new_selected = [x for x in selected if x != drop_idx]
                    new_covered = set()
                    for s in new_selected:
                        new_covered |= all_blocks[s][6]
                    if not (all_blocks[idx][6] & new_covered):
                        new_selected.append(idx)
                        # Check if this is valid (all disjoint)
                        valid = True
                        temp_covered = set()
                        for s in new_selected:
                            if all_blocks[s][6] & temp_covered:
                                valid = False
                                break
                            temp_covered |= all_blocks[s][6]
                        if valid and len(new_selected) > len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break

            # Try pure addition
            if not improved:
                for idx in range(num_blocks):
                    if idx in selected_set:
                        continue
                    if all_blocks[idx][6] & covered:
                        continue
                    selected.append(idx)
                    improved = True
                    break

        return selected

    selected_indices = try_local_search(selected_indices, deadline - 20)

    # Build the decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()

    for idx in selected_indices:
        i0, i1, j0, j1, k0, k1, _ = all_blocks[idx]
        terms = strassen_factors(i0, i1, j0, j1, k0, k1)
        for u_dict, v_dict, w_dict in terms:
            U_cols.append(u_dict)
            V_cols.append(v_dict)
            W_cols.append(w_dict)
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))

    # Pack remaining terms with small blocks
    remaining_terms = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    remaining_terms.add((i, j, k))

    # Generate small blocks
    small_blocks = []

    # 2x2x1 blocks
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k in range(p):
                i1, j1 = i0 + 1, j0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('2x2x1', i0, i1, j0, j1, k, k, entries))

    # 2x1x2 blocks
    for i0 in range(n - 1):
        for j in range(m):
            for k0 in range(p - 1):
                i1, k1 = i0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for k in (k0, k1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('2x1x2', i0, i1, j, j, k0, k1, entries))

    # 1x2x2 blocks
    for i in range(n):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                j1, k1 = j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for j in (j0, j1) for k in (k0, k1))
                if entries.issubset(remaining_terms):
                    small_blocks.append(('1x2x2', i, i, j0, j1, k0, k1, entries))

    # Greedy selection of non-conflicting small blocks
    num_small = len(small_blocks)
    small_conflict = [0] * num_small
    for i in range(num_small):
        for j in range(i + 1, num_small):
            if small_blocks[i][7] & small_blocks[j][7]:
                small_conflict[i] |= (1 << j)
                small_conflict[j] |= (1 << i)

    selected_small = []
    available_small = (1 << num_small) - 1

    while available_small and time.time() < deadline - 5:
        best_idx = -1
        min_conf = float('inf')
        mask = available_small
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            mask ^= lsb
            conf = get_bit_count(small_conflict[idx] & available_small)
            if conf < min_conf:
                min_conf = conf
                best_idx = idx

        if best_idx < 0:
            break

        selected_small.append(best_idx)
        available_small &= ~(1 << best_idx)
        available_small &= ~small_conflict[best_idx]

    # Add small blocks
    for idx in selected_small:
        shape_type, i0, i1, j0, j1, k0, k1, _ = small_blocks[idx]

        if shape_type == '2x2x1':
            a00 = i0 * m + j0
            a01 = i0 * m + j1
            a10 = i1 * m + j0
            a11 = i1 * m + j1
            b00 = j0 * p + k0
            b01 = j1 * p + k0
            c00 = k0 * n + i0
            c01 = k0 * n + i1

            U_cols.append({a00: 1.0, a11: 1.0})
            V_cols.append({b00: 1.0, b01: 1.0})
            W_cols.append({c00: 1.0, c01: 1.0})

            U_cols.append({a01: 1.0, a10: 1.0})
            V_cols.append({b01: -1.0, b00: -1.0})
            W_cols.append({c00: 1.0, c01: 1.0})

            for i in (i0, i1):
                for j in (j0, j1):
                    covered.add((i, j, k0))

        elif shape_type == '2x1x2':
            a00 = i0 * m + j0
            a10 = i1 * m + j0
            b00 = j0 * p + k0
            b01 = j0 * p + k1
            c00 = k0 * n + i0
            c10 = k1 * n + i0
            c01 = k0 * n + i1
            c11 = k1 * n + i1

            U_cols.append({a00: 1.0, a10: 1.0})
            V_cols.append({b00: 1.0})
            W_cols.append({c00: 1.0, c11: 1.0})

            U_cols.append({a00: 1.0})
            V_cols.append({b01: 1.0})
            W_cols.append({c10: 1.0, c01: -1.0})

            for i in (i0, i1):
                for k in (k0, k1):
                    covered.add((i, j0, k))

        elif shape_type == '1x2x2':
            a00 = i0 * m + j0
            a01 = i0 * m + j1
            b00 = j0 * p + k0
            b01 = j0 * p + k1
            b10 = j1 * p + k0
            b11 = j1 * p + k1
            c00 = k0 * n + i0
            c01 = k1 * n + i0

            U_cols.append({a00: 1.0})
            V_cols.append({b00: 1.0, b01: 1.0})
            W_cols.append({c00: 1.0, c01: 1.0})

            U_cols.append({a01: 1.0})
            V_cols.append({b10: 1.0, b11: 1.0})
            W_cols.append({c00: 1.0, c01: -1.0})

            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i0, j, k))

    # Add remaining uncovered terms as schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1.0})
                V_cols.append({j * p + k: 1.0})
                W_cols.append({k * n + i: 1.0})

    rank = len(U_cols)
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val

    # Verify
    if not verify(U, V, W):
        U, V, W, rank = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
