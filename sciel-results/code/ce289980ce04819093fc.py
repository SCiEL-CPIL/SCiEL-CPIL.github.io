import time
import random

def construct():
    deadline = time.time() + 52
    max_val = 256
    max_size = 60

    def validate_and_build(candidates, deadline):
        """Greedily add candidates using incremental bitset DP with dynamic mask."""
        result = []
        sums = [0] * (max_size + 1)
        sums[0] = 1

        for x in candidates:
            if time.time() > deadline:
                break
            if x < 1 or x > 256:
                continue
            if x in result:
                continue

            n = len(result)
            max_sum = (n + 1) * max_val + 1
            mask = (1 << max_sum) - 1

            valid = True
            new_bits_per_k = []
            for k in range(1, n + 2):
                nb = (sums[k - 1] << x) & mask
                new_bits_per_k.append(nb)

            for k in range(1, n + 2):
                nb = new_bits_per_k[k - 1]
                if nb == 0:
                    continue
                for j in range(max_size + 1):
                    if j == k:
                        continue
                    if nb & sums[j]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                result.append(x)
                new_sums = [0] * (max_size + 1)
                new_sums[0] = sums[0]
                for k in range(1, n + 2):
                    new_sums[k] = sums[k] | new_bits_per_k[k - 1]
                sums = new_sums
                if len(result) >= max_size:
                    break

        return result

    def _is_valid_set(s):
        """Check if set s satisfies the cardinality-separated subset sum property."""
        n = len(s)
        max_sum = n * max_val + 1
        mask = (1 << max_sum) - 1
        sums = [0] * (max_size + 1)
        sums[0] = 1
        for val in s:
            new_sums = [0] * (max_size + 1)
            new_sums[0] = sums[0]
            for k in range(1, n + 1):
                new_sums[k] = sums[k] | ((sums[k - 1] << val) & mask)
            for k in range(n + 1):
                for j in range(k + 1, n + 1):
                    if new_sums[k] & new_sums[j]:
                        return False
            sums = new_sums
        return True

    def try_extend(best, deadline):
        """Try to add remaining elements one by one."""
        if len(best) >= max_size:
            return best
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_copy = best[:]
            used = set(best_copy)
            for x in remaining:
                if time.time() > deadline:
                    break
                test_set = best_copy + [x]
                if _is_valid_set(test_set):
                    best_copy.append(x)
                    used.add(x)
                    improved = True
                    if len(best_copy) >= max_size:
                        break
            if improved:
                best = best_copy
                remaining = [x for x in range(1, 257) if x not in used]
        return best

    def local_search(best, deadline):
        """Try swaps that enable further additions."""
        if len(best) >= max_size:
            return best
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline - 2:
                    break
                old_val = best[i]
                for new_val in remaining:
                    if new_val == old_val:
                        continue
                    test_set = best[:]
                    test_set[i] = new_val
                    if _is_valid_set(test_set):
                        extended = try_extend(test_set, deadline)
                        if len(extended) > len(best):
                            best = extended
                            used = set(best)
                            remaining = [x for x in range(1, 257) if x not in used]
                            improved = True
                            break
                if improved:
                    break
        return best

    random.seed(42)
    results = []

    results.append(validate_and_build(range(1, 257), deadline))
    results.append(validate_and_build(range(256, 0, -1), deadline))

    large_first = sorted(range(129, 257), reverse=True) + sorted(range(1, 129))
    results.append(validate_and_build(large_first, deadline))

    for _ in range(40):
        if time.time() > deadline - 5:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        results.append(validate_and_build(candidates, deadline))

    best = max(results, key=len)
    best = try_extend(best, deadline)

    if time.time() < deadline - 3:
        best = local_search(best, deadline)

    best = try_extend(best, deadline)

    if len(best) == 0:
        return [1]

    best = [int(x) for x in best]
    return best
