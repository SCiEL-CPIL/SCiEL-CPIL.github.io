import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Early-terminating determinant check.
        """
        m = len(arr)
        if m < 4:
            return True

        # diff[i] = arr[i] - p
        diff = arr.astype(np.float64) - p.astype(np.float64)
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        sq_diff = (arr.astype(np.float64) ** 2).sum(axis=1) - (p.astype(np.float64) ** 2).sum()

        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff

        # Generate all C(m,4) index combinations
        combo_indices = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
        if len(combo_indices) == 0:
            return True

        # Process in batches for early termination
        batch_size = 10000
        for start in range(0, len(combo_indices), batch_size):
            end = min(start + batch_size, len(combo_indices))
            M_sub = M[combo_indices[start:end]]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
                return False

        return True

    def get_random_order():
        return np.random.permutation(num_points_total)

    def greedy_build(order):
        curr_list = []
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(np.array(curr_list, dtype=np.int64) if curr_list else np.empty((0,3), dtype=np.int64), p):
                curr_list.append(p)
        return np.array(curr_list, dtype=np.int64) if curr_list else np.empty((0, 3), dtype=np.int64)

    best_set = np.empty((0, 3), dtype=np.int64)

    np.random.seed(42)

    # Phase 1: Multiple Greedy Restarts
    num_restarts = 60
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        order = get_random_order()
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search
    current_set = best_set.copy()

    # Build a tuple-to-index map for the grid
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}

    current_indices = set()
    for p in current_set:
        current_indices.add(grid_tuple_to_idx[tuple(p)])

    all_indices = set(range(num_points_total))
    remaining_indices = list(all_indices - current_indices)

    rejection_counts = {}

    no_improve_count = 0
    max_no_improve = 200

    def get_pos_in_set(p_tuple):
        for i in range(len(current_set)):
            if tuple(current_set[i]) == p_tuple:
                return i
        return -1

    while time.time() < deadline:
        improved = False

        # 1. Try to Add
        if remaining_indices:
            # Get candidates sorted by rejection count
            rem_arr = np.array(remaining_indices, dtype=np.int32)
            rej_arr = np.array([rejection_counts.get(idx, 0) for idx in remaining_indices], dtype=np.int32)
            order_idx = np.argsort(rej_arr)
            top_k = min(80, len(rem_arr))
            top_candidates = rem_arr[order_idx[:top_k]].tolist()

            # Add some random ones
            if len(remaining_indices) > top_k:
                extra = set(remaining_indices) - set(top_candidates)
                if extra:
                    extra_list = list(extra)
                    num_extra = min(15, len(extra_list))
                    top_candidates.extend(np.random.choice(extra_list, num_extra, replace=False).tolist())

            np.random.shuffle(top_candidates)

            for idx in top_candidates:
                p = grid[idx]
                if is_valid_addition(current_set, p):
                    current_set = np.vstack([current_set, p]) if len(current_set) > 0 else p[None, :]
                    current_indices.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    rejection_counts[idx] = rejection_counts.get(idx, 0) + 1

        # 2. Try to Swap
        if not improved and len(current_indices) > 0 and remaining_indices:
            swaps_to_try = 25
            for _ in range(swaps_to_try):
                if time.time() > deadline:
                    break

                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]

                # Pick out: prioritize high rejection
                rem_arr = np.array(remaining_indices, dtype=np.int32)
                rej_arr = np.array([rejection_counts.get(idx, 0) for idx in remaining_indices], dtype=np.int32)
                order_idx = np.argsort(rej_arr)[::-1]
                top_k = min(15, len(rem_arr))
                top_out = rem_arr[order_idx[:top_k]].tolist()

                out_idx_val = top_out[np.random.randint(len(top_out))]
                p_out = grid[out_idx_val]

                in_pos = get_pos_in_set(tuple(grid[in_idx_val]))
                if in_pos == -1:
                    continue

                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]

                if is_valid_addition(reduced_set, p_out):
                    current_set = reduced_set
                    if len(current_set) == 0:
                        current_set = p_out[None, :]
                    else:
                        current_set = np.vstack([current_set, p_out])

                    current_indices.remove(in_idx_val)
                    current_indices.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx_val)

                    rejection_counts[out_idx_val] = 0
                    rejection_counts[in_idx_val] = rejection_counts.get(in_idx_val, 0) + 1

                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if len(current_indices) > 10:
                    kick_size = min(5, max(3, len(current_indices) // 6))
                    kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                    for idx in kick_idxs:
                        pos = get_pos_in_set(tuple(grid[idx]))
                        if pos != -1:
                            mask = np.ones(len(current_set), dtype=bool)
                            mask[pos] = False
                            current_set = current_set[mask]
                        current_indices.remove(idx)
                        remaining_indices.append(idx)
                        rejection_counts[idx] = 0
                    no_improve_count = 0
                    max_no_improve = 300

    return current_set.astype(np.int64)
