# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    N = 512
    MAXC = 48

    def _deadline_exceeded():
        return time.time() >= DEADLINE

    def _is_mean_free(s):
        """
        Exact check: no element a in s equals the arithmetic mean of any subset
        of size >= 2 from s \\ {a}.
        Uses integer bitset DP for subset sums of exact cardinalities.
        Returns True if valid.
        """
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

    def _cantor_set(max_val):
        """
        Generate the Cantor-like set: numbers in [1, max_val] whose base-3
        representation uses only digits 0 and 1.
        This set is 3-AP-free.
        """
        result = []
        digits = 0
        temp = max_val
        while temp > 0:
            digits += 1
            temp //= 3
        for mask in range(1, 1 << digits):
            val = 0
            for d in range(digits):
                if mask & (1 << d):
                    val += (3 ** d)
            if 1 <= val <= max_val:
                result.append(val)
        result.sort()
        return result

    def _greedy_add(s, s_set, candidates):
        """
        Greedily add candidates to s, checking full mean-freeness each time.
        """
        result = list(s)
        result_set = set(s_set)
        remaining = [x for x in candidates if x not in result_set]

        while len(result) < MAXC and remaining and not _deadline_exceeded():
            best_x = None
            best_score = None

            for x in remaining:
                # Quick 2-AP conflict check
                has_2ap = False
                for a in result:
                    b = 2 * x - a
                    if b in result_set and b != a and b != x:
                        has_2ap = True
                        break
                    b2 = 2 * a - x
                    if b2 in result_set and b2 != a and b2 != x:
                        has_2ap = True
                        break
                if has_2ap:
                    continue

                # Heuristic: prefer elements far from existing ones
                score = sum(1 for a in result if abs(a - x) <= max(1, x // 2))
                if best_score is None or score < best_score:
                    best_score = score
                    best_x = x
                    if score == 0:
                        break

            if best_x is None:
                # No candidate passed 2-AP check; try full validation on all
                for x in remaining:
                    test_s = result + [x]
                    if _is_mean_free(test_s):
                        best_x = x
                        break
                if best_x is None:
                    break

            test_s = result + [best_x]
            if _is_mean_free(test_s):
                result.append(best_x)
                result_set.add(best_x)
                remaining.remove(best_x)
            else:
                remaining.remove(best_x)

        return result, result_set

    def _local_search_add(s, s_set):
        """Try to add any element from 1..N that isn't in s, maintaining validity."""
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for x in range(1, N + 1):
                if _deadline_exceeded():
                    break
                if x in s_set:
                    continue
                if len(s) >= MAXC:
                    break
                ok = True
                for a in s:
                    b = 2 * x - a
                    if b in s_set and b != a and b != x:
                        ok = False
                        break
                    b2 = 2 * a - x
                    if b2 in s_set and b2 != a and b2 != x:
                        ok = False
                        break
                if not ok:
                    continue
                test_s = s + [x]
                if _is_mean_free(test_s):
                    s = test_s
                    s_set.add(x)
                    improved = True
                    break
        return s, s_set

    def _local_search_swap(s, s_set):
        """Try to replace one element with another to maintain/increase size."""
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for i in range(len(s)):
                if _deadline_exceeded():
                    break
                old = s[i]
                for x in range(1, N + 1):
                    if _deadline_exceeded():
                        break
                    if x in s_set:
                        continue
                    new_s = [v for j, v in enumerate(s) if j != i] + [x]
                    new_s.sort()
                    if len(new_s) <= len(s):
                        continue
                    ok = True
                    new_set = set(new_s)
                    for a in new_s:
                        for b in new_s:
                            if a == b:
                                continue
                            c = 2 * a - b
                            if c in new_set and c != a and c != b:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    if _is_mean_free(new_s):
                        s = new_s
                        s_set = set(s)
                        improved = True
                        break
                if improved:
                    break
        return s, s_set

    # Phase 1: Generate Cantor set (3-AP-free base, ~64 elements in [1,512])
    cantor = _cantor_set(N)

    # Phase 2: Start with Cantor set
    best = sorted(cantor)
    best_set = set(best)

    # Phase 3: Validate Cantor set; repair if needed
    if not _is_mean_free(best):
        while not _is_mean_free(best) and not _deadline_exceeded():
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
                        best_set.discard(a)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break

    # Phase 4: Greedy add from all candidates
    all_candidates = list(range(1, N + 1))
    best, best_set = _greedy_add(best, best_set, all_candidates)

    # Phase 5: Local search - add
    best, best_set = _local_search_add(best, best_set)

    # Phase 6: Local search - swaps
    if not _deadline_exceeded():
        best, best_set = _local_search_swap(best, best_set)

    # Final validation
    if not _is_mean_free(best):
        while not _is_mean_free(best) and not _deadline_exceeded():
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
                        best_set.discard(a)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break

    return best

# EVOLVE-BLOCK-END
