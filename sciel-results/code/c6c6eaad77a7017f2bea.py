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
    max_code = int(base[0])  # n^(n-1)

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

    def get_window_code(arr, start):
        """Compute code for window starting at position start in arr."""
        window = arr[start:start + n].astype(np.int64)
        return int(window @ base)

    def try_deletion_fast(arr, L_val, pcount, pos):
        """Check if position pos can be deleted. Returns True if safe, else False."""
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False

        # Collect old window codes that will be destroyed
        old_codes = set()
        for w in range(w_start, w_end + 1):
            oc = get_window_code(arr, w)
            old_codes.add(oc)

        # Collect new window codes that will be created
        new_codes = set()
        for j in range(max(0, pos - n_val + 1), pos):
            # After deletion, window starting at j in new array corresponds to:
            # arr[j:pos] + arr[pos+1:j+n]
            if j + n_val > L_val:
                continue
            window = np.concatenate([arr[j:pos], arr[pos + 1:j + n_val]])
            if len(window) == n_val:
                nc = int(window.astype(np.int64) @ base)
                new_codes.add(nc)

        # Check: for each old code with count 1, it must appear in new_codes
        for oc in old_codes:
            if pcount.get(oc, 0) == 1:
                if oc not in new_codes:
                    return False

        return True

    def perform_deletion(arr, L_val, pcount, codes_arr, pos):
        """Perform deletion at pos, update all state. Returns (new_arr, new_L, new_pcount, new_codes)."""
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        # Update perm_count for old windows
        for w in range(w_start, w_end + 1):
            oc = int(codes_arr[w])
            pcount[oc] -= 1
            if pcount[oc] == 0:
                del pcount[oc]

        # Update perm_count for new windows
        for j in range(max(0, pos - n_val + 1), pos):
            if j + n_val > L_val:
                continue
            window = np.concatenate([arr[j:pos], arr[pos + 1:j + n_val]])
            if len(window) == n_val:
                nc = int(window.astype(np.int64) @ base)
                pcount[nc] = pcount.get(nc, 0) + 1

        # Perform the deletion
        new_arr = np.concatenate([arr[:pos], arr[pos + 1:]])
        new_L = L_val - 1
        new_codes = compute_all_codes(new_arr, new_L)

        return new_arr, new_L, pcount, new_codes

    # Phase 1: Repeated deletion passes
    while time.time() < deadline - 3.0:
        deleted_this_pass = False

        # Try deleting from right to left first
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 3.0:
                break

            if try_deletion_fast(current, L, perm_count, pos):
                current, L, perm_count, codes = perform_deletion(current, L, perm_count, codes, pos)
                best_len = min(best_len, L)
                deleted_this_pass = True
                # Don't break - continue scanning for more deletions
                continue

        if not deleted_this_pass:
            # Try left to right
            for pos in range(L):
                if time.time() > deadline - 3.0:
                    break

                if try_deletion_fast(current, L, perm_count, pos):
                    current, L, perm_count, codes = perform_deletion(current, L, perm_count, codes, pos)
                    best_len = min(best_len, L)
                    deleted_this_pass = True
                    continue

        if not deleted_this_pass:
            break

    # Phase 2: 2-opt style moves - delete and reinsert
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        rng = np.random.default_rng(42)

        # Precompute all valid permutation codes
        all_perm_codes = set()
        for perm in permutations(range(1, n + 1)):
            code = int(np.array(perm, dtype=np.int64) @ base)
            all_perm_codes.add(code)

        # Count how many times each required code appears
        # We need to track which codes are "critical" (appear exactly once)
        critical_codes = set()
        for code in all_perm_codes:
            if perm_count.get(code, 0) == 1:
                critical_codes.add(code)

        max_rounds = int(remaining_time / 0.5)

        for round_num in range(max_rounds):
            if time.time() > deadline - 2.0:
                break

            # Random position to try deleting
            pos = int(rng.integers(0, L))
            symbol = current[pos]

            if not try_deletion_fast(current, L, perm_count, pos):
                continue

            # Save state
            saved_current = current.copy()
            saved_L = L
            saved_codes = codes.copy()
            saved_perm_count = dict(perm_count)

            # Perform deletion
            current, L, perm_count, codes = perform_deletion(current, L, perm_count, codes, pos)

            # Try to find a valid insertion position
            best_insert_pos = -1
            best_insert_valid = False

            # Try positions near the original position first, then randomly
            candidate_positions = list(range(max(0, pos - 5), min(L + 1, pos + 6)))
            # Add some random positions
            for _ in range(10):
                rp = int(rng.integers(0, L + 1))
                if rp not in candidate_positions:
                    candidate_positions.append(rp)

            for insert_pos in candidate_positions:
                if time.time() > deadline - 2.0:
                    break

                # Build new array with symbol inserted at insert_pos
                new_arr = np.concatenate([current[:insert_pos], np.array([symbol], dtype=np.int32), current[insert_pos:]])
                new_L = L + 1

                # Compute new codes and check validity
                new_codes_arr = compute_all_codes(new_arr, new_L)
                new_pcount = compute_counts_from_codes(new_codes_arr)

                # Check if all permutations are still covered
                if len(new_pcount) >= target_perms:
                    # Verify all required codes are present
                    valid = True
                    for ac in all_perm_codes:
                        if ac not in new_pcount:
                            valid = False
                            break
                    if valid:
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
                # Restore state
                current = saved_current
                L = saved_L
                codes = saved_codes
                perm_count = saved_perm_count

    # Final phase: one more deletion pass with the improved state
    if time.time() < deadline - 1.0:
        deleted = True
        while deleted and time.time() < deadline - 1.0:
            deleted = False
            for pos in range(L - 1, -1, -1):
                if time.time() > deadline - 1.0:
                    break
                if try_deletion_fast(current, L, perm_count, pos):
                    current, L, perm_count, codes = perform_deletion(current, L, perm_count, codes, pos)
                    best_len = min(best_len, L)
                    deleted = True
                    break

    return "".join(map(str, current.tolist()))
