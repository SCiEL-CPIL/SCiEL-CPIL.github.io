import time
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # 1. Generate initial superpermutation
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

    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

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
    all_perm_codes = set()
    for perm in permutations(range(1, n + 1)):
        code = int(np.array(perm, dtype=np.int64) @ base)
        all_perm_codes.add(code)

    def try_deletion(arr, L_val, codes_arr, pcount, pos):
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False, None

        old_codes = codes_arr[w_start:w_end + 1]
        
        new_codes_list = []
        for j in range(w_start, pos):
            if j + n_val > L_val:
                break
            new_window = np.concatenate([arr[j:pos], arr[pos + 1:j + n_val]])
            code = int(new_window.astype(np.int64) @ base)
            new_codes_list.append(code)

        new_codes_set = set(new_codes_list)

        for oc in old_codes:
            oc_int = int(oc)
            if pcount.get(oc_int, 0) == 1:
                if oc_int not in new_codes_set:
                    return False, None

        return True, new_codes_list

    def try_swap(arr, L_val, codes_arr, pcount, pos1, pos2):
        # Swap arr[pos1] and arr[pos2]
        # This affects windows overlapping pos1 or pos2
        # For simplicity, we can just build the new array and check validity
        # But that is O(L*n).
        # A faster check:
        # The swap only changes windows that contain pos1 or pos2.
        # If |pos1 - pos2| >= n, the affected windows are disjoint.
        # If |pos1 - pos2| < n, they overlap.
        
        # Given the complexity of an incremental check for swap, 
        # and that swaps are only used as a "kick" to escape local minima,
        # we can afford a slightly slower check, or just do it on the full array 
        # but limit the number of swap attempts.
        
        # Let's try a simple approach:
        # Perform the swap on a copy, compute codes, check if all perms present.
        # This is O(L*n) which is ~40k operations. 
        # If we do this 1000 times, it's 40M ops, which is fine in 150s.
        
        new_arr = arr.copy()
        new_arr[pos1], new_arr[pos2] = new_arr[pos2], new_arr[pos1]
        
        new_codes = compute_all_codes(new_arr, L_val)
        new_pcount = compute_counts_from_codes(new_codes)
        
        if len(new_pcount) < target_perms:
            return False, None
        
        # Check if all required perms are present
        for pc in all_perm_codes:
            if pc not in new_pcount:
                return False, None
                
        return True, new_arr

    # Main search loop
    while time.time() < deadline:
        deleted_this_pass = False
        swapped_this_pass = False
        
        # 1. Try deletions
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break

            can_del, new_codes_list = try_deletion(current, L, codes, perm_count, pos)

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

        if deleted_this_pass:
            continue # Restart scan

        # 2. If no deletion, try swaps
        if L > 1:
            rng = np.random.default_rng(42)
            # Try a few random swaps
            for _ in range(100):
                if time.time() > deadline - 2.0:
                    break
                
                p1 = int(rng.integers(0, L))
                p2 = int(rng.integers(0, L))
                if p1 == p2:
                    continue
                
                can_swap, new_arr = try_swap(current, L, codes, perm_count, p1, p2)
                if can_swap:
                    current = new_arr
                    codes = compute_all_codes(current, L)
                    perm_count = compute_counts_from_codes(codes)
                    swapped_this_pass = True
                    break
                    
        if not swapped_this_pass:
            break

    return "".join(map(str, current.tolist()))
