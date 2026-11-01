import time
import numpy as np
from collections import defaultdict
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin below 150s

    # 1. Generate initial superpermutation using standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search with vectorized deletion checks
    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

    # Use a base that fits in int64 safely for n=7
    base = np.array([n ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    def compute_all_codes(arr, L_val):
        if L_val < n:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, n)
        return windows.astype(np.int64) @ base

    def compute_counts_from_codes(codes):
        if len(codes) == 0:
            return {}
        unique, counts = np.unique(codes, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    codes = compute_all_codes(current, L)
    perm_count = compute_counts_from_codes(codes)

    if len(perm_count) < target_perms:
        return "".join(map(str, current.tolist()))

    best_len = L

    # Precompute all valid permutation codes for quick lookup
    all_perm_codes = set()
    for perm in permutations(range(1, n + 1)):
        code = int(np.array(perm, dtype=np.int64) @ base)
        all_perm_codes.add(code)

    def try_deletion_fast(arr, L_val, codes_arr, pcount, pos):
        """Vectorized deletion check. Returns (can_del, new_codes_list) or (False, None)."""
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False, None

        old_codes = codes_arr[w_start:w_end + 1]

        # Build new windows that form after deleting position pos
        # Window starting at j (j from w_start to pos-1) becomes arr[j:pos] + arr[pos+1:j+n_val]
        new_codes_list = []
        for j in range(w_start, pos):
            if j + n_val > L_val:
                break
            # The new window is arr[j:pos] concatenated with arr[pos+1:j+n_val]
            new_window = np.concatenate([arr[j:pos], arr[pos + 1:j + n_val]])
            code = int(new_window.astype(np.int64) @ base)
            new_codes_list.append(code)

        new_codes_set = set(new_codes_list)

        # Check: for each old code with count 1, it must appear in new_codes_set
        for oc in old_codes:
            oc_int = int(oc)
            if pcount.get(oc_int, 0) == 1:
                if oc_int not in new_codes_set:
                    return False, None

        return True, new_codes_list

    # Phase 1: Repeated deletion passes (right-to-left then left-to-right)
    while time.time() < deadline:
        deleted_this_pass = False

        # Try deleting from right to left
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break

            can_del, new_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)

            if can_del:
                w_start = max(0, pos - n + 1)
                w_end = min(pos, L - n)
                old_codes = codes[w_start:w_end + 1]
                for oc in old_codes:
                    oc_int = int(oc)
                    perm_count[oc_int] -= 1
                    if perm_count[oc_int] == 0:
                        del perm_count[oc_int]

                for nc in new_codes_list:
                    perm_count[nc] = perm_count.get(nc, 0) + 1

                current = np.concatenate([current[:pos], current[pos + 1:]])
                L -= 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                deleted_this_pass = True
                break

        if not deleted_this_pass:
            # Try deleting from left to right
            for pos in range(L):
                if time.time() > deadline - 2.0:
                    break

                can_del, new_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)

                if can_del:
                    w_start = max(0, pos - n + 1)
                    w_end = min(pos, L - n)
                    old_codes = codes[w_start:w_end + 1]
                    for oc in old_codes:
                        oc_int = int(oc)
                        perm_count[oc_int] -= 1
                        if perm_count[oc_int] == 0:
                            del perm_count[oc_int]

                    for nc in new_codes_list:
                        perm_count[nc] = perm_count.get(nc, 0) + 1

                    current = np.concatenate([current[:pos], current[pos + 1:]])
                    L -= 1
                    best_len = min(best_len, L)
                    codes = compute_all_codes(current, L)
                    deleted_this_pass = True
                    break

        if not deleted_this_pass:
            break

    # Phase 2: 2-opt style moves (delete and reinsert)
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        rng = np.random.default_rng(42)
        # Budget: each 2-opt attempt takes roughly O(L*n) time, so limit attempts
        max_2opt_rounds = int(remaining_time / 5.0)

        for round_num in range(max_2opt_rounds):
            if time.time() > deadline - 1.0:
                break

            pos = int(rng.integers(0, L))
            symbol = int(current[pos])

            can_del, new_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)
            if not can_del:
                continue

            saved_current = current.copy()
            saved_L = L
            saved_codes = codes.copy()
            saved_perm_count = dict(perm_count)

            # Perform deletion
            w_start = max(0, pos - n + 1)
            w_end = min(pos, L - n)
            old_codes = codes[w_start:w_end + 1]
            for oc in old_codes:
                oc_int = int(oc)
                perm_count[oc_int] -= 1
                if perm_count[oc_int] == 0:
                    del perm_count[oc_int]

            for nc in new_codes_list:
                perm_count[nc] = perm_count.get(nc, 0) + 1

            current = np.concatenate([current[:pos], current[pos + 1:]])
            L -= 1
            codes = compute_all_codes(current, L)

            # Try inserting symbol at various positions to see if we can get shorter
            # Only try a subset of positions for speed
            best_insert_pos = -1
            best_insert_valid = False

            # Try positions near the original position first, then spread out
            candidate_positions = [pos, pos - 1, pos + 1, pos - 2, pos + 2]
            candidate_positions = [p for p in candidate_positions if 0 <= p <= L]
            # Add some random positions
            if L > 0:
                for _ in range(min(5, L)):
                    candidate_positions.append(int(rng.integers(0, L + 1)))

            for insert_pos in candidate_positions:
                if time.time() > deadline - 1.0:
                    break
                if not (0 <= insert_pos <= L):
                    continue

                # Build new array with symbol inserted at insert_pos
                new_string = np.concatenate([current[:insert_pos], np.array([symbol], dtype=np.int32), current[insert_pos:]])
                new_L = L + 1
                new_codes_arr = compute_all_codes(new_string, new_L)
                new_pcount = compute_counts_from_codes(new_codes_arr)

                # Check if all permutations are present
                if len(new_pcount) >= target_perms:
                    # Verify all required codes are present
                    all_present = True
                    for pc in all_perm_codes:
                        if pc not in new_pcount:
                            all_present = False
                            break
                    if all_present:
                        best_insert_pos = insert_pos
                        best_insert_valid = True
                        break

            if best_insert_valid:
                current = np.concatenate([current[:best_insert_pos], np.array([symbol], dtype=np.int32), current[best_insert_pos:]])
                L = L + 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                perm_count = compute_counts_from_codes(codes)
            else:
                # Restore
                current = saved_current
                L = saved_L
                codes = saved_codes
                perm_count = saved_perm_count

    return "".join(map(str, current.tolist()))
