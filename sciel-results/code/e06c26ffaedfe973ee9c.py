# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0
    N = 512
    MAXC = 48

    def _deadline():
        return time.time() >= DEADLINE

    def _is_mean_free(s):
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            a = s[i]
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k - 1] << v)
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    return False
        return True

    def _can_add(x, current):
        """Check if x can be added to current set while maintaining mean-free property.
        Returns True if valid."""
        if not current:
            return True
        # Check 1: is x the mean of any subset of current (size >= 2)?
        n_cur = len(current)
        if n_cur >= 2:
            dp = [0] * (n_cur + 1)
            dp[0] = 1
            for v in current:
                for k in range(n_cur, 0, -1):
                    dp[k] |= (dp[k - 1] << v)
            for k in range(2, n_cur + 1):
                if (dp[k] >> (k * x)) & 1:
                    return False
        # Check 2: for each a in current, does adding x make a the mean of a subset?
        # a is mean of subset S of (current\{a}) ∪ {x}, |S| >= 2
        # Case A: x not in S → already checked by _is_mean_free(current) assumption
        # Case B: x in S → S = T ∪ {x}, T ⊆ current\{a}, |T| >= 1
        #   sum(T) + x = (|T|+1) * a  →  sum(T) = (|T|+1)*a - x
        for a in current:
            others = [v for v in current if v != a]
            n_o = len(others)
            if n_o < 1:
                continue
            dp2 = [0] * (n_o + 1)
            dp2[0] = 1
            for v in others:
                for k in range(n_o, 0, -1):
                    dp2[k] |= (dp2[k - 1] << v)
            for j in range(1, n_o + 1):
                target = (j + 1) * a - x
                if 0 <= target and (dp2[j] >> target) & 1:
                    return False
        return True

    def _greedy_build(ordering, initial=None):
        """Build set greedily following ordering, starting from initial if given."""
        selected = list(initial) if initial else []
        sel_set = set(selected)
        for x in ordering:
            if _deadline():
                break
            if len(selected) >= MAXC:
                break
            if x in sel_set:
                continue
            if _can_add(x, selected):
                selected.append(x)
                sel_set.add(x)
        return selected

    def _local_search_add(s):
        s_set = set(s)
        improved = True
        while improved and not _deadline():
            improved = False
            for x in range(1, N + 1):
                if _deadline():
                    break
                if x in s_set:
                    continue
                if len(s) >= MAXC:
                    break
                if _can_add(x, s):
                    s.append(x)
                    s_set.add(x)
                    improved = True
                    break
        return s

    def _local_search_swap(s):
        """Try removing one element and adding one or two new ones."""
        s_set = set(s)
        improved = True
        while improved and not _deadline():
            improved = False
            for i in range(len(s) - 1, -1, -1):
                if _deadline():
                    break
                if len(s) >= MAXC:
                    break
                old = s[i]
                # Try to remove old and add a new element
                for x in range(1, N + 1):
                    if _deadline():
                        break
                    if x in s_set:
                        continue
                    if len(s) >= MAXC:
                        break
                    # Temporarily remove old
                    s.pop(i)
                    s_set.discard(old)
                    if _can_add(x, s):
                        s.append(x)
                        s_set.add(x)
                        improved = True
                        break
                    else:
                        # Restore
                        s.insert(i, old)
                        s_set.add(old)
                if improved:
                    break
        return s

    # Cantor base-3 digits 0/1 numbers: these are structurally 3-AP-free
    cantor = []
    v = 1
    while v <= N and len(cantor) < MAXC:
        cantor.append(v)
        # Next Cantor number: multiply by 3 (skip the middle third)
        v *= 3
    # cantor = [1, 3, 9, 27, 81, 243] -> 6 elements

    # Also include some small values to start
    base_candidates = sorted(set(cantor + list(range(1, 10))))

    # Multiple orderings
    orderings = []

    # 1. Cantor first, then rest ascending
    rest1 = [x for x in range(1, N + 1) if x not in set(cantor)]
    orderings.append(cantor + rest1)

    # 2. Ascending
    orderings.append(list(range(1, N + 1)))

    # 3. Descending
    orderings.append(list(range(N, 0, -1)))

    # 4. Small first, then large
    orderings.append(list(range(1, N // 2 + 1)) + list(range(N, N // 2, -1)))

    # 5. Odd first, then even
    orderings.append([x for x in range(1, N + 1) if x % 2 == 1] +
                      [x for x in range(1, N + 1) if x % 2 == 0])

    # 6. Not divisible by 3 first
    orderings.append([x for x in range(1, N + 1) if x % 3 != 0] +
                      [x for x in range(1, N + 1) if x % 3 == 0])

    # 7-12. Random shuffles
    rng = random.Random(42)
    for seed in range(6):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # Put Cantor numbers at front
        cantor_set = set(cantor)
        front = [x for x in order if x in cantor_set]
        back = [x for x in order if x not in cantor_set]
        orderings.append(front + back)

    best = []
    for ordering in orderings:
        if _deadline():
            break
        # Try with and without initial base
        result = _greedy_build(ordering)
        if len(result) > len(best):
            best = result
        if len(best) >= MAXC:
            break

    # Local search on best
    if len(best) < MAXC and not _deadline():
        best = _local_search_add(best)

    if len(best) < MAXC and not _deadline():
        best = _local_search_swap(best)

    # Final validation and repair
    if not _is_mean_free(best):
        while not _is_mean_free(best) and not _deadline():
            n = len(best)
            removed = False
            for i in range(n):
                a = best[i]
                dp = [0] * (n + 1)
                dp[0] = 1
                for j in range(n):
                    if j == i:
                        continue
                    v = best[j]
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k - 1] << v)
                for k in range(2, n):
                    if (dp[k] >> (k * a)) & 1:
                        best.pop(i)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break

    # Final attempt to fill
    if len(best) < MAXC and not _deadline():
        best = _local_search_add(best)

    return best

# EVOLVE-BLOCK-END
