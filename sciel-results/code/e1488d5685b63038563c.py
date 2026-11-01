import time
import numpy as np
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)

    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
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

    # 2. Local Search Deletion with NumPy-accelerated counting
    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

    # Precompute base for code computation
    base = np.array([n ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    def compute_window_codes(arr, L_val, n_val):
        """Compute all window codes as a numpy array."""
        if L_val < n_val:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, n_val)
        codes = windows.astype(np.int64) @ base
        return codes

    def compute_perm_count(codes):
        """Compute permutation counts from codes array."""
        if len(codes) == 0:
            return {}
        unique, counts = np.unique(codes, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    window_codes = compute_window_codes(current, L, n)
    perm_count = compute_perm_count(window_codes)

    # If we don't have all perms, return as is
    if len(perm_count) < target_perms:
        return "".join(map(str, current.tolist()))

    def try_delete(i, current_arr, L_val, window_codes_arr, perm_count_dict):
        """Try to delete position i. Returns (success, new_current, new_codes, new_count)."""
        n_val = n
        # Old windows containing position i: start indices from max(0, i-n+1) to min(i, L-n)
        start_w = max(0, i - n_val + 1)
        end_w = min(i, L_val - n_val)

        if start_w > end_w:
            # No windows contain this position (shouldn't happen for valid positions)
            return False, None, None, None

        old_codes = window_codes_arr[start_w:end_w + 1]

        # Check if any old code has count 1 and won't survive
        # First, compute new window codes that cross the deletion point
        new_codes_list = []
        for j in range(max(0, i - n_val + 1), i):
            prefix_len = i - j
            suffix_len = n_val - prefix_len
            if i + 1 + suffix_len > L_val:
                continue
            if prefix_len > 0:
                prefix = current_arr[j:i]
            else:
                prefix = np.array([], dtype=np.int32)
            if suffix_len > 0:
                suffix = current_arr[i + 1:i + 1 + suffix_len]
            else:
                suffix = np.array([], dtype=np.int32)
            window = np.concatenate([prefix, suffix]).astype(np.int64)
            code = int(window @ base)
            new_codes_list.append(code)

        new_codes_set = set(new_codes_list)

        # Check: for each old window with count 1, it must be in new_codes_set
        for code in old_codes:
            code_int = int(code)
            if perm_count_dict.get(code_int, 0) == 1:
                if code_int not in new_codes_set:
                    return False, None, None, None

        # Perform deletion
        # Update perm_count
        new_count = dict(perm_count_dict)
        for code in old_codes:
            code_int = int(code)
            new_count[code_int] = new_count.get(code_int, 0) - 1
            if new_count[code_int] == 0:
                del new_count[code_int]

        for code in new_codes_list:
            code_int = int(code)
            new_count[code_int] = new_count.get(code_int, 0) + 1

        new_current = np.concatenate([current_arr[:i], current_arr[i + 1:]])
        L_new = L_val - 1
        new_window_codes = compute_window_codes(new_current, L_new, n_val)

        return True, new_current, new_window_codes, new_count

    # Fast deletion loop using precomputed window codes
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline - 10:
            break

        deleted_any = False
        # Alternate scan direction
        if pass_num % 2 == 0:
            indices = list(range(L - 1, -1, -1))
        else:
            indices = list(range(L))

        for i in indices:
            if time.time() > deadline - 10:
                break
            if i >= L:
                continue

            success, new_current, new_codes, new_count = try_delete(
                i, current, L, window_codes, perm_count
            )

            if success:
                current = new_current
                window_codes = new_codes
                perm_count = new_count
                L -= 1
                deleted_any = True
                break  # Restart scan after deletion

        if not deleted_any:
            break  # No more deletions possible

    # 3. Additional local search: try swaps to enable more deletions
    if time.time() < deadline - 20:
        for swap_pass in range(20):
            if time.time() > deadline - 15:
                break

            improvement = False
            # Try random swaps
            num_swaps = min(100, L)
            for _ in range(num_swaps):
                if time.time() > deadline - 15:
                    break

                i = np.random.randint(0, L)
                j = np.random.randint(0, L)
                if i == j:
                    continue

                # Try swapping positions i and j
                old_current = current.copy()
                current[i], current[j] = current[j], current[i]

                new_codes = compute_window_codes(current, L, n)
                new_count = compute_perm_count(new_codes)

                # Check validity
                if len(new_count) >= target_perms:
                    # Count how many deletions are now possible (rough heuristic)
                    # Just accept if valid
                    window_codes = new_codes
                    perm_count = new_count
                    improvement = True
                else:
                    # Revert
                    current = old_current

            if not improvement:
                break

    # 4. Final deletion pass
    if time.time() < deadline - 5:
        for final_pass in range(30):
            if time.time() > deadline - 5:
                break

            deleted_any = False
            if final_pass % 2 == 0:
                indices = list(range(L - 1, -1, -1))
            else:
                indices = list(range(L))

            for i in indices:
                if time.time() > deadline - 5:
                    break
                if i >= L:
                    continue

                success, new_current, new_codes, new_count = try_delete(
                    i, current, L, window_codes, perm_count
                )

                if success:
                    current = new_current
                    window_codes = new_codes
                    perm_count = new_count
                    L -= 1
                    deleted_any = True
                    break

            if not deleted_any:
                break

    return "".join(map(str, current.tolist()))
