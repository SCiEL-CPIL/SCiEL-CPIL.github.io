# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590

    def generate_vectors(n_nonzero):
        """Generate all vectors with exactly n_nonzero non-zero entries (each ±1)."""
        vecs = []
        for combo in itertools.combinations(range(d), n_nonzero):
            for signs in itertools.product([1, -1], repeat=n_nonzero):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs

    # Generate candidates by norm level as numpy arrays
    candidates_2 = np.array(generate_vectors(2), dtype=np.int64)
    candidates_4 = np.array(generate_vectors(4), dtype=np.int64)
    candidates_6 = np.array(generate_vectors(6), dtype=np.int64)
    candidates_8 = np.array(generate_vectors(8), dtype=np.int64)
    candidates_10 = np.array(generate_vectors(10), dtype=np.int64)

    norm_to_candidates = {
        2: (candidates_2, 2),
        4: (candidates_4, 4),
        6: (candidates_6, 6),
        8: (candidates_8, 8),
        10: (candidates_10, 10),
    }

    def greedy_select_vec(cand_arr, norm_sq, existing_arr, existing_norms, max_norm):
        """
        Vectorized greedy select.
        cand_arr: (n_cands, d) int64
        existing_arr: (n_exist, d) int64
        existing_norms: (n_exist,) int64
        max_norm: int
        Returns: (selected_indices, selected_vectors_list)
        """
        n_cands = cand_arr.shape[0]
        selected_idx = []
        selected_vecs = []

        # Precompute dot products with existing
        if existing_arr.shape[0] > 0:
            dots_exist = cand_arr @ existing_arr.T  # (n_cands, n_exist)
        else:
            dots_exist = None

        for i in range(n_cands):
            if time.time() > deadline - 3:
                break
            v = cand_arr[i]
            valid = True

            if existing_arr.shape[0] > 0:
                dots_row = dots_exist[i]
                allowed = (existing_norms + norm_sq - max_norm) / 2.0
                if np.any(dots_row > allowed):
                    valid = False

            if valid and len(selected_vecs) > 0:
                sel_tmp = np.array(selected_vecs, dtype=np.int64)
                dots_sel = sel_tmp @ v
                allowed_sel = (2.0 * norm_sq - max_norm) / 2.0
                if np.any(dots_sel > allowed_sel):
                    valid = False

            if valid:
                selected_idx.append(i)
                selected_vecs.append(v)

        return selected_idx, selected_vecs

    def validate_and_count(points):
        """Validate the set and return count if valid, else 0."""
        if len(points) < 2:
            return 0
        arr = np.array(points, dtype=np.int64)
        norms = np.sum(arr * arr, axis=1)
        max_norm = int(np.max(norms))
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[i] - arr[j]
                sq_dist = int(np.sum(diff * diff))
                if sq_dist < max_norm:
                    return 0
        return n

    def try_add(arr, norms, max_norm, v, v_norm):
        """Try adding v to set. Returns (ok, new_arr, new_norms, new_max)."""
        new_max = max(max_norm, v_norm)
        if arr.shape[0] > 0:
            dots = arr @ v
            allowed = (norms + v_norm - new_max) / 2.0
            if np.any(dots > allowed):
                return False, arr, norms, max_norm
        new_arr = np.vstack([arr, v.reshape(1, -1)])
        new_norms = np.append(norms, v_norm)
        return True, new_arr, new_norms, new_max

    def local_improve(points, max_norm_init, time_limit=12.0):
        """Local search: try adding points, then try remove-one-and-refill."""
        if len(points) < 2:
            return points
        arr = np.array(points, dtype=np.int64)
        norms = np.sum(arr * arr, axis=1)
        cur_max = max(max_norm_init, int(np.max(norms)))
        n0 = len(arr)
        t0 = time.time()

        # Phase 1: greedily add more points
        for nl in [2, 4, 6, 8, 10]:
            if time.time() > t0 + time_limit:
                break
            cands, nsq = norm_to_candidates[nl]
            for i in range(cands.shape[0]):
                if time.time() > t0 + time_limit:
                    break
                v = cands[i]
                if np.any(np.all(arr == v, axis=1)):
                    continue
                ok, arr, norms, cur_max = try_add(arr, norms, cur_max, v, nsq)
                if ok:
                    pass

        best_arr = arr
        best_norms = norms
        best_max = cur_max
        best_n = arr.shape[0]

        # Phase 2: try removing one point and refilling
        for rem in range(best_n):
            if time.time() > t0 + time_limit:
                break
            rem_arr = np.delete(best_arr, rem, axis=0)
            rem_norms = np.delete(best_norms, rem)
            rem_max = int(np.max(rem_norms)) if len(rem_norms) > 0 else 0
            added = 0
            for nl in [2, 4, 6, 8, 10]:
                if time.time() > t0 + time_limit:
                    break
                cands, nsq = norm_to_candidates[nl]
                for i in range(cands.shape[0]):
                    if time.time() > t0 + time_limit:
                        break
                    v = cands[i]
                    if np.any(np.all(rem_arr == v, axis=1)):
                        continue
                    ok, rem_arr, rem_norms, rem_max = try_add(rem_arr, rem_norms, rem_max, v, nsq)
                    if ok:
                        added += 1
                        if added > 1:
                            break
                if added > 1:
                    break
            if rem_arr.shape[0] > best_n:
                best_n = rem_arr.shape[0]
                best_arr = rem_arr
                best_norms = rem_norms
                best_max = rem_max

        if best_n > n0:
            return list(best_arr)
        return points

    best_points = []
    best_count = 0

    strategies = [
        (4, [4, 6, 2]),
        (4, [4, 2, 6]),
        (4, [6, 4, 2]),
        (4, [2, 4, 6]),
        (6, [4, 6, 2]),
        (6, [6, 4, 2]),
        (6, [4, 2, 6]),
        (6, [2, 4, 6]),
        (6, [6, 2, 4]),
        (8, [4, 6, 8, 2]),
        (8, [6, 4, 8, 2]),
        (8, [8, 4, 6, 2]),
        (10, [4, 6, 8, 10, 2]),
        (10, [6, 4, 8, 10, 2]),
        (2, [2, 4, 6]),
        (2, [4, 2, 6]),
    ]

    for max_norm, norm_order in strategies:
        if time.time() > deadline - 80:
            break
        valid_order = [n for n in norm_order if n <= max_norm]
        if not valid_order:
            continue
        existing_arr = np.zeros((0, d), dtype=np.int64)
        existing_norms = np.zeros(0, dtype=np.int64)

        for norm_level in valid_order:
            if time.time() > deadline - 40:
                break
            cands, nsq = norm_to_candidates[norm_level]
            _, sel_vecs = greedy_select_vec(cands, nsq, existing_arr, existing_norms, max_norm)
            if sel_vecs:
                existing_arr = np.vstack([existing_arr, np.array(sel_vecs, dtype=np.int64)])
                existing_norms = np.concatenate([existing_norms, np.array([nsq] * len(sel_vecs), dtype=np.int64)])

        cnt = validate_and_count(list(existing_arr))
        if cnt > best_count:
            best_count = cnt
            best_points = list(existing_arr)

    # Randomized restarts
    rng = random.Random(42)
    for restart in range(15):
        if time.time() > deadline - 60:
            break
        max_norm = rng.choice([4, 6, 8])
        norm_levels = [n for n in [2, 4, 6, 8, 10] if n <= max_norm]
        rng.shuffle(norm_levels)

        existing_arr = np.zeros((0, d), dtype=np.int64)
        existing_norms = np.zeros(0, dtype=np.int64)

        for norm_level in norm_levels:
            if time.time() > deadline - 30:
                break
            cands, nsq = norm_to_candidates[norm_level]
            order = np.random.default_rng(restart * 100 + norm_level).permutation(cands.shape[0])
            shuffled_cands = cands[order]
            _, sel_vecs = greedy_select_vec(shuffled_cands, nsq, existing_arr, existing_norms, max_norm)
            if sel_vecs:
                existing_arr = np.vstack([existing_arr, np.array(sel_vecs, dtype=np.int64)])
                existing_norms = np.concatenate([existing_norms, np.array([nsq] * len(sel_vecs), dtype=np.int64)])

        if time.time() < deadline - 20:
            improved = local_improve(list(existing_arr), max_norm, time_limit=10.0)
            if len(improved) > len(existing_arr):
                existing_arr = np.array(improved, dtype=np.int64)

        cnt = validate_and_count(list(existing_arr))
        if cnt > best_count:
            best_count = cnt
            best_points = list(existing_arr)

    # Final local improvement on best
    if time.time() < deadline - 15 and best_count > 0:
        improved = local_improve(best_points, 8, time_limit=20.0)
        if len(improved) > best_count:
            cnt = validate_and_count(improved)
            if cnt > best_count:
                best_count = cnt
                best_points = improved

    if not best_points:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1; points.append(p1)
                p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1; points.append(p2)
        return np.array(points, dtype=np.int64)

    result = np.array(best_points, dtype=np.int64)
    return np.around(result).astype(np.int64)

# EVOLVE-BLOCK-END
