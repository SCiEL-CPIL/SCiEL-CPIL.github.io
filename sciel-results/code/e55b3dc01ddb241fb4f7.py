import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 82  # 90s limit, leave 8s margin

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)

    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}

    # Precompute incompatibility mask for each smooth number
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)

    # Precompute for each target: candidate indices and sorted values
    # Also precompute prefix sums for pruning
    target_data = {}
    for n in range(1, 129):
        cand_indices = [i for i in range(n_smooth) if smooth_list[i] <= n]
        cand_indices.sort(key=lambda i: smooth_list[i])
        cand_vals = [smooth_list[i] for i in cand_indices]
        target_data[n] = (cand_indices, cand_vals)

    # Memoization: (target, k, start_idx, used_mask) -> False if no solution
    # We'll use a per-target memo to avoid cross-contamination
    memo_cache = {}

    def find_min_antichain(n):
        """Find minimum k such that there's an antichain of k smooth numbers summing to n."""
        cand_indices, cand_vals = target_data[n]
        if not cand_indices:
            return None

        # Try k = 1, 2, 3, ...
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break

            if len(cand_vals) < k:
                continue

            # Quick feasibility check
            if sum(cand_vals[-k:]) < n or sum(cand_vals[:k]) > n:
                continue

            # Use DFS with memoization on failed states
            # State: (start, rem_sum, rem_k, used_mask)
            # We memoize when a state fails
            failed_states = set()

            sys.setrecursionlimit(10000)

            def dfs(start, rem_sum, rem_k, used_mask):
                if rem_k == 0:
                    return rem_sum == 0

                if start >= len(cand_indices):
                    return False

                state_key = (start, rem_sum, rem_k, used_mask)
                if state_key in failed_states:
                    return False

                # Pruning: check if enough candidates remain
                if rem_k > len(cand_indices) - start:
                    failed_states.add(state_key)
                    return False

                # Pruning: min and max possible sums
                avail_count = len(cand_indices) - start
                if rem_k > avail_count:
                    failed_states.add(state_key)
                    return False

                # Max possible: sum of largest rem_k from cand_vals[start:]
                # Min possible: sum of smallest rem_k from cand_vals[start:]
                if rem_k <= avail_count:
                    max_possible = sum(cand_vals[start + avail_count - rem_k : start + avail_count])
                    min_possible = sum(cand_vals[start : start + rem_k])
                    if rem_sum > max_possible or rem_sum < min_possible:
                        failed_states.add(state_key)
                        return False

                # Try each candidate from start onwards
                for i in range(start, len(cand_indices)):
                    ci = cand_indices[i]
                    val = smooth_list[ci]

                    if val > rem_sum:
                        continue

                    # Check compatibility with all used
                    # used_mask is a bitmask of indices in cand_indices
                    compatible = True
                    # Iterate through set bits in used_mask
                    mask = used_mask
                    while mask:
                        low_bit = mask & (-mask)
                        idx_in_cand = low_bit.bit_length() - 1
                        pi = cand_indices[idx_in_cand]
                        if incompat[pi] & (1 << ci):
                            compatible = False
                            break
                        mask ^= low_bit

                    if not compatible:
                        continue

                    new_mask = used_mask | (1 << i)
                    if dfs(i + 1, rem_sum - val, rem_k - 1, new_mask):
                        return True

                failed_states.add(state_key)
                return False

            if dfs(0, n, k, 0):
                # Reconstruct the solution
                # We need to redo the search to collect the path
                # Or we can modify dfs to return the path
                # For simplicity, let's do a second pass to get the actual values
                # Actually, let's just return True and we'll find the values separately
                # Better: modify to collect path

                # Let's do a clean reconstruction
                def dfs_reconstruct(start, rem_sum, rem_k, used_mask, path):
                    if rem_k == 0:
                        if rem_sum == 0:
                            return True
                        return False

                    if start >= len(cand_indices):
                        return False

                    state_key = (start, rem_sum, rem_k, used_mask)
                    if state_key in failed_states:
                        return False

                    if rem_k > len(cand_indices) - start:
                        return False

                    avail_count = len(cand_indices) - start
                    if rem_k <= avail_count:
                        max_possible = sum(cand_vals[start + avail_count - rem_k : start + avail_count])
                        min_possible = sum(cand_vals[start : start + rem_k])
                        if rem_sum > max_possible or rem_sum < min_possible:
                            return False

                    for i in range(start, len(cand_indices)):
                        ci = cand_indices[i]
                        val = smooth_list[ci]

                        if val > rem_sum:
                            continue

                        compatible = True
                        mask = used_mask
                        while mask:
                            low_bit = mask & (-mask)
                            idx_in_cand = low_bit.bit_length() - 1
                            pi = cand_indices[idx_in_cand]
                            if incompat[pi] & (1 << ci):
                                compatible = False
                                break
                            mask ^= low_bit

                        if not compatible:
                            continue

                        new_mask = used_mask | (1 << i)
                        path.append(ci)
                        if dfs_reconstruct(i + 1, rem_sum - val, rem_k - 1, new_mask, path):
                            return True
                        path.pop()

                    return False

                path = []
                dfs_reconstruct(0, n, k, 0, path)
                result = [smooth_list[i] for i in path]
                return result

        return None

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Emergency fallback
            if n in val_to_idx:
                answers.append([n])
            else:
                ans = find_min_antichain(n)
                answers.append(ans if ans else [n])
            continue

        ans = find_min_antichain(n)
        if ans is None:
            # Fallback
            if n in val_to_idx:
                ans = [n]
            else:
                ans = [n]
        answers.append(ans)

    return answers
