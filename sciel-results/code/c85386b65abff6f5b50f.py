import time
import random

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    random.seed(42)

    def valid_set(A):
        return len(set(A)) == 20 and all(1 <= x <= 10000 for x in A)

    # --- Lattice-based construction: minimize |A*A| via exponent structure ---
    # Represent elements as 2^i * 3^j. Products become 2^(i1+i2) * 3^(j1+j2).
    # Dense 2D grids in exponent space give small product sets.

    lattice_candidates = []

    # Dense 2D grids
    for i_max in range(0, 14):
        for j_max in range(0, 10):
            points = [(i, j) for i in range(i_max + 1) for j in range(j_max + 1)]
            if len(points) >= 20:
                points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
                subset = points_sorted[:20]
                vals = [(2 ** i) * (3 ** j) for (i, j) in subset]
                if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                    lattice_candidates.append(sorted(vals))

    # Structured lines in the lattice
    for c in range(0, 25):
        points = []
        for j in range(0, 10):
            i = c - 2 * j
            if 0 <= i <= 14:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = [(2 ** i) * (3 ** j) for (i, j) in subset]
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))

    for c in range(0, 30):
        points = []
        for i in range(0, 14):
            j = c - 2 * i
            if 0 <= j <= 10:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = [(2 ** i) * (3 ** j) for (i, j) in subset]
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))

    # Deduplicate lattice candidates
    seen = set()
    unique_lattice = []
    for cand in lattice_candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique_lattice.append(cand)

    # --- Additional hand-crafted and GP candidates ---
    hand_crafted = [
        list(range(1, 21)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36],
    ]

    for base in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]:
        A = [int(round(base ** i)) for i in range(20)]
        seen_g = set()
        uniq = []
        for x in A:
            if 1 <= x <= 10000 and x not in seen_g:
                seen_g.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        if len(uniq) == 20 and all(1 <= x <= 10000 for x in uniq):
            hand_crafted.append(uniq)

    all_candidates = unique_lattice + hand_crafted

    # Evaluate all initial candidates
    best_set = None
    best_obj = float('inf')
    for cand in all_candidates:
        if not valid_set(cand):
            continue
        obj = objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_set = cand[:]

    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)

    # --- Fast local search with incremental set updates ---

    def fast_local_search(A_init):
        nonlocal best_set, best_obj
        A = list(A_init)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]

        # Precompute initial sum and product sets with counts
        def compute_sets(arr):
            sums = {}
            prods = {}
            n = len(arr)
            for i in range(n):
                for j in range(i, n):
                    s = arr[i] + arr[j]
                    sums[s] = sums.get(s, 0) + 1
                    p = arr[i] * arr[j]
                    prods[p] = prods.get(p, 0) + 1
            return sums, prods

        sums_dict, prods_dict = compute_sets(A)

        def try_replace(i, new_val):
            """Try replacing A[i] with new_val. Returns new objective or None if invalid."""
            nonlocal sums_dict, prods_dict
            old_val = A[i]
            if new_val == old_val or new_val < 1 or new_val > 10000:
                return None
            # Check if new_val already in A (excluding position i)
            for k in range(20):
                if k != i and A[k] == new_val:
                    return None

            # Remove old pairs involving position i
            n = 20
            for j in range(n):
                if j == i:
                    continue
                s_old = old_val + A[j]
                p_old = old_val * A[j]
                sums_dict[s_old] -= 1
                if sums_dict[s_old] == 0:
                    del sums_dict[s_old]
                prods_dict[p_old] -= 1
                if prods_dict[p_old] == 0:
                    del prods_dict[p_old]
            # Remove diagonal pair (i,i)
            s_diag = old_val + old_val
            p_diag = old_val * old_val
            sums_dict[s_diag] -= 1
            if sums_dict[s_diag] == 0:
                del sums_dict[s_diag]
            prods_dict[p_diag] -= 1
            if prods_dict[p_diag] == 0:
                del prods_dict[p_diag]

            # Add new pairs involving position i
            A[i] = new_val
            for j in range(n):
                if j == i:
                    continue
                s_new = new_val + A[j]
                p_new = new_val * A[j]
                sums_dict[s_new] = sums_dict.get(s_new, 0) + 1
                prods_dict[p_new] = prods_dict.get(p_new, 0) + 1
            # Add diagonal pair
            s_diag_new = new_val + new_val
            p_diag_new = new_val * new_val
            sums_dict[s_diag_new] = sums_dict.get(s_diag_new, 0) + 1
            prods_dict[p_diag_new] = prods_dict.get(p_diag_new, 0) + 1

            new_obj = max(len(sums_dict), len(prods_dict))
            return new_obj

        def undo_replace(i, old_val, new_val):
            """Undo a replacement."""
            nonlocal sums_dict, prods_dict
            n = 20
            # Remove new pairs
            for j in range(n):
                if j == i:
                    continue
                s_new = new_val + A[j]
                p_new = new_val * A[j]
                sums_dict[s_new] -= 1
                if sums_dict[s_new] == 0:
                    del sums_dict[s_new]
                prods_dict[p_new] -= 1
                if prods_dict[p_new] == 0:
                    del prods_dict[p_new]
            s_diag_new = new_val + new_val
            p_diag_new = new_val * new_val
            sums_dict[s_diag_new] -= 1
            if sums_dict[s_diag_new] == 0:
                del sums_dict[s_diag_new]
            prods_dict[p_diag_new] -= 1
            if prods_dict[p_diag_new] == 0:
                del prods_dict[p_diag_new]

            # Restore old pairs
            A[i] = old_val
            for j in range(n):
                if j == i:
                    continue
                s_old = old_val + A[j]
                p_old = old_val * A[j]
                sums_dict[s_old] = sums_dict.get(s_old, 0) + 1
                prods_dict[p_old] = prods_dict.get(p_old, 0) + 1
            s_diag = old_val + old_val
            p_diag = old_val * old_val
            sums_dict[s_diag] = sums_dict.get(s_diag, 0) + 1
            prods_dict[p_diag] = prods_dict.get(p_diag, 0) + 1

        def try_swap(i, j):
            """Try swapping A[i] and A[j]. Returns new objective or None."""
            nonlocal sums_dict, prods_dict
            vi = A[i]
            vj = A[j]
            if vi == vj:
                return None

            # Remove all pairs involving i or j
            n = 20
            for k in range(n):
                if k == i or k == j:
                    continue
                # Pair (i, k)
                s_ik = vi + A[k]
                p_ik = vi * A[k]
                sums_dict[s_ik] -= 1
                if sums_dict[s_ik] == 0:
                    del sums_dict[s_ik]
                prods_dict[p_ik] -= 1
                if prods_dict[p_ik] == 0:
                    del prods_dict[p_ik]
                # Pair (j, k)
                s_jk = vj + A[k]
                p_jk = vj * A[k]
                sums_dict[s_jk] -= 1
                if sums_dict[s_jk] == 0:
                    del sums_dict[s_jk]
                prods_dict[p_jk] -= 1
                if prods_dict[p_jk] == 0:
                    del prods_dict[p_jk]
            # Pair (i, j)
            s_ij = vi + vj
            p_ij = vi * vj
            sums_dict[s_ij] -= 1
            if sums_dict[s_ij] == 0:
                del sums_dict[s_ij]
            prods_dict[p_ij] -= 1
            if prods_dict[p_ij] == 0:
                del prods_dict[p_ij]
            # Diagonal (i,i)
            s_ii = vi + vi
            p_ii = vi * vi
            sums_dict[s_ii] -= 1
            if sums_dict[s_ii] == 0:
                del sums_dict[s_ii]
            prods_dict[p_ii] -= 1
            if prods_dict[p_ii] == 0:
                del prods_dict[p_ii]
            # Diagonal (j,j)
            s_jj = vj + vj
            p_jj = vj * vj
            sums_dict[s_jj] -= 1
            if sums_dict[s_jj] == 0:
                del sums_dict[s_jj]
            prods_dict[p_jj] -= 1
            if prods_dict[p_jj] == 0:
                del prods_dict[p_jj]

            # Perform swap
            A[i] = vj
            A[j] = vi

            # Add new pairs
            for k in range(n):
                if k == i or k == j:
                    continue
                # Pair (i, k) with new A[i]=vj
                s_ik = vj + A[k]
                p_ik = vj * A[k]
                sums_dict[s_ik] = sums_dict.get(s_ik, 0) + 1
                prods_dict[p_ik] = prods_dict.get(p_ik, 0) + 1
                # Pair (j, k) with new A[j]=vi
                s_jk = vi + A[k]
                p_jk = vi * A[k]
                sums_dict[s_jk] = sums_dict.get(s_jk, 0) + 1
                prods_dict[p_jk] = prods_dict.get(p_jk, 0) + 1
            # Pair (i, j)
            s_ij_new = vj + vi
            p_ij_new = vj * vi
            sums_dict[s_ij_new] = sums_dict.get(s_ij_new, 0) + 1
            prods_dict[p_ij_new] = prods_dict.get(p_ij_new, 0) + 1
            # Diagonal (i,i)
            s_ii_new = vj + vj
            p_ii_new = vj * vj
            sums_dict[s_ii_new] = sums_dict.get(s_ii_new, 0) + 1
            prods_dict[p_ii_new] = prods_dict.get(p_ii_new, 0) + 1
            # Diagonal (j,j)
            s_jj_new = vi + vi
            p_jj_new = vi * vi
            sums_dict[s_jj_new] = sums_dict.get(s_jj_new, 0) + 1
            prods_dict[p_jj_new] = prods_dict.get(p_jj_new, 0) + 1

            new_obj = max(len(sums_dict), len(prods_dict))
            return new_obj

        def undo_swap(i, j, vi, vj):
            """Undo a swap."""
            nonlocal sums_dict, prods_dict
            n = 20
            # Remove new pairs (same as adding old pairs back)
            for k in range(n):
                if k == i or k == j:
                    continue
                s_ik = vj + A[k]
                p_ik = vj * A[k]
                sums_dict[s_ik] -= 1
                if sums_dict[s_ik] == 0:
                    del sums_dict[s_ik]
                prods_dict[p_ik] -= 1
                if prods_dict[p_ik] == 0:
                    del prods_dict[p_ik]
                s_jk = vi + A[k]
                p_jk = vi * A[k]
                sums_dict[s_jk] -= 1
                if sums_dict[s_jk] == 0:
                    del sums_dict[s_jk]
                prods_dict[p_jk] -= 1
                if prods_dict[p_jk] == 0:
                    del prods_dict[p_jk]
            s_ij_new = vj + vi
            p_ij_new = vj * vi
            sums_dict[s_ij_new] -= 1
            if sums_dict[s_ij_new] == 0:
                del sums_dict[s_ij_new]
            prods_dict[p_ij_new] -= 1
            if prods_dict[p_ij_new] == 0:
                del prods_dict[p_ij_new]
            s_ii_new = vj + vj
            p_ii_new = vj * vj
            sums_dict[s_ii_new] -= 1
            if sums_dict[s_ii_new] == 0:
                del sums_dict[s_ii_new]
            prods_dict[p_ii_new] -= 1
            if prods_dict[p_ii_new] == 0:
                del prods_dict[p_ii_new]
            s_jj_new = vi + vi
            p_jj_new = vi * vi
            sums_dict[s_jj_new] -= 1
            if sums_dict[s_jj_new] == 0:
                del sums_dict[s_jj_new]
            prods_dict[p_jj_new] -= 1
            if prods_dict[p_jj_new] == 0:
                del prods_dict[p_jj_new]

            # Restore original
            A[i] = vi
            A[j] = vj

            # Add old pairs back
            for k in range(n):
                if k == i or k == j:
                    continue
                s_ik = vi + A[k]
                p_ik = vi * A[k]
                sums_dict[s_ik] = sums_dict.get(s_ik, 0) + 1
                prods_dict[p_ik] = prods_dict.get(p_ik, 0) + 1
                s_jk = vj + A[k]
                p_jk = vj * A[k]
                sums_dict[s_jk] = sums_dict.get(s_jk, 0) + 1
                prods_dict[p_jk] = prods_dict.get(p_jk, 0) + 1
            s_ij = vi + vj
            p_ij = vi * vj
            sums_dict[s_ij] = sums_dict.get(s_ij, 0) + 1
            prods_dict[p_ij] = prods_dict.get(p_ij, 0) + 1
            s_ii = vi + vi
            p_ii = vi * vi
            sums_dict[s_ii] = sums_dict.get(s_ii, 0) + 1
            prods_dict[p_ii] = prods_dict.get(p_ii, 0) + 1
            s_jj = vj + vj
            p_jj = vj * vj
            sums_dict[s_jj] = sums_dict.get(s_jj, 0) + 1
            prods_dict[p_jj] = prods_dict.get(p_jj, 0) + 1

        # Main local search loop
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False

            # Phase 1: Try all single replacements
            for i in range(20):
                if time.time() > deadline - 3:
                    break
                old_val = A[i]
                best_cand_val = old_val
                best_cand_obj = cur_obj

                # Generate candidates
                cands = set()
                # Additive neighbors
                for d in range(-5, 6):
                    cands.add(old_val + d)
                # Multiplicative neighbors
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0, 3.0]:
                    cands.add(int(round(old_val * factor)))
                # Lattice-inspired
                for mult in [2, 3, 5]:
                    cands.add(old_val * mult)
                    if old_val % mult == 0:
                        cands.add(old_val // mult)
                # Random
                cands.add(random.randint(1, 10000))
                cands.add(random.randint(1, 100))

                for cand in cands:
                    if cand < 1 or cand > 10000 or cand == old_val:
                        continue
                    # Check uniqueness
                    dup = False
                    for k in range(20):
                        if k != i and A[k] == cand:
                            dup = True
                            break
                    if dup:
                        continue

                    new_obj = try_replace(i, cand)
                    if new_obj is not None and new_obj < best_cand_obj:
                        best_cand_obj = new_obj
                        best_cand_val = cand
                    undo_replace(i, old_val, cand)

                if best_cand_val != old_val:
                    A[i] = best_cand_val
                    cur_obj = best_cand_obj
                    improved = True
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_set = A[:]

            # Phase 2: Try swaps to escape local minima
            if time.time() < deadline - 3:
                for i in range(20):
                    if time.time() > deadline - 3:
                        break
                    for j in range(i + 1, 20):
                        if time.time() > deadline - 3:
                            break
                        vi = A[i]
                        vj = A[j]
                        if vi == vj:
                            continue
                        new_obj = try_swap(i, j)
                        if new_obj is not None and new_obj < cur_obj:
                            cur_obj = new_obj
                            improved = True
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best_set = A[:]
                        else:
                            undo_swap(i, j, vi, vj)

        return A

    # Run local search from the best seed
    if time.time() < deadline - 10:
        fast_local_search(best_set)

    # Run from a few other good seeds if time permits
    for cand in all_candidates[:5]:
        if time.time() > deadline - 8:
            break
        if tuple(cand) == tuple(best_set):
            continue
        fast_local_search(cand)

    # Final polish
    if time.time() < deadline - 2:
        fast_local_search(best_set)

    return best_set
