"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)

    # Only divisors <= target are usable
    usable = [d for d in divisors if d <= target]
    usable.sort(reverse=True)
    m = len(usable)

    # Precompute prefix sums for fast range sum queries
    # prefix[i] = sum(usable[0:i])
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + usable[i]

    def range_sum(i, j):
        """Sum of usable[i:j] where i <= j"""
        if i >= j:
            return 0
        return prefix[j] - prefix[i]

    # Try k = 1, 2, 3, ... with DFS and strong pruning
    def dfs(start_idx, remaining, depth, k, path):
        if time.time() > DEADLINE:
            return None
        if depth == k:
            if remaining == 0:
                return list(path)
            return None

        terms_left = k - depth

        # Not enough divisors left
        if m - start_idx < terms_left:
            return None

        # Max sum: take the largest terms_left divisors from start_idx onward
        max_possible = range_sum(start_idx, start_idx + terms_left)
        if max_possible < remaining:
            return None

        # Min sum: take the smallest terms_left divisors from start_idx onward
        # The smallest terms_left divisors in usable[start_idx:m] are the last terms_left elements
        min_start = m - terms_left
        if min_start < start_idx:
            min_start = start_idx
        min_possible = range_sum(min_start, m)
        if min_possible > remaining:
            return None

        for i in range(start_idx, m):
            d = usable[i]
            if d > remaining:
                # Since usable is sorted descending, all subsequent are smaller
                # But we need d <= remaining, so continue
                continue

            new_remaining = remaining - d
            new_terms_left = terms_left - 1

            if new_terms_left == 0:
                if new_remaining == 0:
                    path.append(d)
                    result = list(path)
                    path.pop()
                    return result
                continue

            # Check if enough divisors remain
            if m - (i + 1) < new_terms_left:
                break

            # Max possible after picking d: largest new_terms_left from usable[i+1:m]
            max_after = range_sum(i + 1, i + 1 + new_terms_left)
            if max_after < new_remaining:
                break  # subsequent picks will only make max_after smaller

            # Min possible after picking d: smallest new_terms_left from usable[i+1:m]
            min_start_after = m - new_terms_left
            if min_start_after < i + 1:
                min_start_after = i + 1
            min_after = range_sum(min_start_after, m)
            if min_after > new_remaining:
                # Since usable is descending, as i increases, the available range shrinks
                # and the minimum possible sum stays the same or increases
                # Actually, as i increases, we're excluding more large elements,
                # so the minimum sum from the remaining set stays the same (still the smallest elements)
                # until we've excluded all but fewer than new_terms_left elements
                # So this doesn't necessarily break, but it's a valid prune
                continue

            path.append(d)
            result = dfs(i + 1, new_remaining, depth + 1, k, path)
            if result is not None:
                return result
            path.pop()

        return None

    best_result = None
    for k in range(1, 12):
        if time.time() > DEADLINE:
            break
        result = dfs(0, target, 0, k, [])
        if result is not None:
            best_result = result
            break

    if best_result is None:
        # Fallback to greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        best_result = chosen

    # Verify
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    for d in best_result:
        assert n % d == 0
    assert len(best_result) <= 256

    return best_result
# EVOLVE-BLOCK-END
