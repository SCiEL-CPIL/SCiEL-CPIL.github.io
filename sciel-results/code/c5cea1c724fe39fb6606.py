import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)

    smooth.sort()
    n_smooth = len(smooth)

    # Precompute conflict masks
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and (smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0):
                conflict_mask[i] |= (1 << j)

    # Precompute prefix sums
    prefix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth):
        prefix_sum[i + 1] = prefix_sum[i] + smooth[i]

    def find_min_antichain(target):
        for k in range(1, 31):
            if time.time() > deadline:
                return None

            if k == 1:
                if target in smooth:
                    return [target]
                continue

            min_possible = prefix_sum[k]
            if min_possible > target:
                break

            max_possible = prefix_sum[n_smooth] - prefix_sum[n_smooth - k]
            if max_possible < target:
                continue

            memo = {}

            def dfs(remaining, k_left, start_idx, mask):
                if remaining < 0 or k_left < 0:
                    return None
                if k_left == 0:
                    return () if remaining == 0 else None
                if start_idx + k_left > n_smooth:
                    return None

                min_sum = prefix_sum[start_idx + k_left] - prefix_sum[start_idx]
                if min_sum > remaining:
                    return None

                max_sum = prefix_sum[n_smooth] - prefix_sum[n_smooth - k_left]
                if max_sum < remaining:
                    return None

                key = (remaining, k_left, start_idx, mask)
                if key in memo:
                    return memo[key]

                for i in range(start_idx, n_smooth):
                    val = smooth[i]
                    if val > remaining:
                        break
                    if mask & conflict_mask[i]:
                        continue
                    new_mask = mask | (1 << i)
                    res = dfs(remaining - val, k_left - 1, i + 1, new_mask)
                    if res is not None:
                        memo[key] = (i,) + res
                        return (i,) + res

                memo[key] = None
                return None

            result_indices = dfs(target, k, 0, 0)
            if result_indices is not None:
                return [smooth[i] for i in result_indices]

        return None

    answers = [None] * 129

    for n in range(1, 129):
        if time.time() > deadline:
            break
        res = find_min_antichain(n)
        if res is not None:
            answers[n] = res
        elif n in smooth:
            answers[n] = [n]

    # Fallback for any remaining None
    for n in range(1, 129):
        if answers[n] is None:
            if n in smooth:
                answers[n] = [n]
            else:
                # Try a simpler search without memo to avoid memory issues
                found = False
                for k in range(1, 31):
                    if time.time() > deadline:
                        break
                    def dfs_simple(remaining, k_left, start_idx, mask):
                        if remaining < 0 or k_left < 0:
                            return None
                        if k_left == 0:
                            return () if remaining == 0 else None
                        if start_idx + k_left > n_smooth:
                            return None
                        min_sum = prefix_sum[start_idx + k_left] - prefix_sum[start_idx]
                        if min_sum > remaining:
                            return None
                        for i in range(start_idx, n_smooth):
                            val = smooth[i]
                            if val > remaining:
                                break
                            if mask & conflict_mask[i]:
                                continue
                            res = dfs_simple(remaining - val, k_left - 1, i + 1, mask | (1 << i))
                            if res is not None:
                                return (i,) + res
                        return None
                    res = dfs_simple(n, k, 0, 0)
                    if res is not None:
                        answers[n] = [smooth[i] for i in res]
                        found = True
                        break
                if not found:
                    # Last resort: just use [n] if smooth, else leave as is
                    # But all targets should have solutions
                    if n in smooth:
                        answers[n] = [n]

    return [answers[n] for n in range(1, 129)]
