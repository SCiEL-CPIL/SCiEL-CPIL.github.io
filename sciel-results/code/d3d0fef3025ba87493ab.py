"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import copy

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit

    best_set = []
    best_len = 0

    def try_greedy(order, max_size=48):
        """Try to build a valid set by greedily adding elements in the given order."""
        selected = []
        n = 0
        MAX_SUM = 48 * 512  # 24576

        # dp_all[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements
        dp_all = [1]  # dp_all[0] has bit 0 set (empty subset, sum 0)
        # dp_without[a] is a list of bitmasks for sums achievable with k elements from selected excluding a
        dp_without = {}

        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break

            # Check 1: Is x the mean of any subset of selected of size >= 2?
            # Need: exists k >= 2, subset of selected with sum = k*x
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break

            if not valid:
                continue

            # Check 2: For each a in selected, is a the mean of a subset involving x?
            # Need: exists a in selected, exists k >= 1, subset U of (selected \ {a}) with
            # sum(U) = (k+1)*a - x
            for a in selected:
                if not valid:
                    break
                # dp_without[a][k] gives achievable sums with k elements from selected\{a}
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > MAX_SUM:
                        continue
                    if (dp_without[a][k] >> target) & 1:
                        valid = False
                        break

            if not valid:
                continue

            # Add x to selected
            selected.append(x)

            # Update dp_all: new_dp_all[k] = dp_all[k] | (dp_all[k-1] << x)
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all

            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:  # all except x
                old_dp_w = dp_without[a]
                # new dp_w[k] = old_dp_w[k] | (old_dp_w[k-1] << x)
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list

            # For new element x, dp_without[x] is the old dp_all (selected before adding x)
            new_dp_without[x] = dp_all[:n + 1] + [0] * max(0, n + 2 - (n + 1))

            dp_without = new_dp_without
            n += 1

        return selected

    def try_local_improve(initial_set, order_remaining, max_size=48):
        """Try to improve a set by adding elements from remaining in different orders."""
        if len(initial_set) >= max_size:
            return initial_set

        selected = list(initial_set)
        n = len(selected)
        MAX_SUM = 48 * 512

        # Rebuild DP from the initial set
        dp_all = [0] * (n + 1)
        dp_all[0] = 1
        for val in selected:
            for k in range(n, 0, -1):
                if k - 1 >= 0:
                    dp_all[k] |= (dp_all[k - 1] << val)

        dp_without = {}
        for i, a in enumerate(selected):
            # Build dp for selected without a
            dp_w = [0] * (n)
            dp_w[0] = 1
            for val in selected:
                if val == a:
                    continue
                for k in range(n - 1, 0, -1):
                    if k - 1 >= 0:
                        dp_w[k] |= (dp_w[k - 1] << val)
            dp_without[a] = dp_w + [0] * max(0, n + 1 - (n))

        # Try adding elements from order_remaining
        for x in order_remaining:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break

            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break

            if not valid:
                continue

            for a in selected:
                if not valid:
                    break
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > MAX_SUM:
                        continue
                    if (dp_without[a][k] >> target) & 1:
                        valid = False
                        break

            if not valid:
                continue

            # Add x
            selected.append(x)
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all

            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list

            new_dp_without[x] = dp_all[:n + 1] + [0] * max(0, n + 2 - (n + 1))
            dp_without = new_dp_without
            n += 1

        return selected

    # Strategy 1: Ascending order
    res = try_greedy(range(1, 513))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Descending order
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(512, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Spread ordering (alternate low and high)
    if time.time() - start_time < deadline - 5:
        candidates = list(range(1, 513))
        spread = []
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            spread.append(candidates[lo])
            if lo != hi:
                spread.append(candidates[hi])
            lo += 1
            hi -= 1
        res = try_greedy(spread)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 4: Gap-based ordering (prefer numbers with large gaps to multiples)
    if time.time() - start_time < deadline - 5:
        # Score each number by how "far" it is from being a mean
        # Numbers that are not close to multiples of other numbers are better
        def gap_score(x):
            score = 0
            for y in range(1, 513):
                if y == x:
                    continue
                # Check if x is close to mean of {y, something}
                # For pairs: x = (y + z)/2 => z = 2x - y
                z = 2 * x - y
                if 1 <= z <= 512 and z != x:
                    # x is mean of {y, z}
                    score -= 1
            return score

        candidates = list(range(1, 513))
        candidates.sort(key=lambda x: (-gap_score(x), x))  # most "isolated" first
        res = try_greedy(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 5: Multiple randomized restarts with diverse seeds
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 3:
        elapsed = time.time() - start_time
        time_per_attempt = elapsed / 4 if elapsed > 0 else 10

        max_attempts = max(1, int(remaining_time / max(time_per_attempt, 1)))
        max_attempts = min(max_attempts, 60)  # Cap at 60 attempts

        for i in range(max_attempts):
            if time.time() - start_time > deadline - 2:
                break

            random.seed(i * 7919 + 104729)  # Different primes for variety
            candidates = list(range(1, 513))

            # Different shuffling strategies
            if i % 4 == 0:
                random.shuffle(candidates)
            elif i % 4 == 1:
                # Sort by random hash
                candidates.sort(key=lambda x: hash((x, i)))
            elif i % 4 == 2:
                # Weighted shuffle: prefer mid-range numbers
                weights = [1.0 + 0.5 * (1 - abs(x - 256) / 256) for x in candidates]
                # Partial shuffle with bias
                shuffled = list(candidates)
                for _ in range(10):
                    idx = random.choices(range(len(shuffled)), weights=weights)[0]
                    other = random.randint(0, len(shuffled) - 1)
                    shuffled[idx], shuffled[other] = shuffled[other], shuffled[idx]
                candidates = shuffled
            else:
                # Random subset then shuffle
                subset_size = random.randint(200, 512)
                subset = random.sample(candidates, subset_size)
                random.shuffle(subset)
                candidates = subset

            res = try_greedy(candidates)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)
                if best_len >= 48:
                    break

    # Strategy 6: Local improvement - try to add more elements to best set
    if best_len < 48 and time.time() - start_time < deadline - 3:
        used = set(best_set)
        remaining = [x for x in range(1, 513) if x not in used]
        random.seed(42)
        random.shuffle(remaining)
        improved = try_local_improve(best_set, remaining)
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)

    return best_set

# EVOLVE-BLOCK-END
