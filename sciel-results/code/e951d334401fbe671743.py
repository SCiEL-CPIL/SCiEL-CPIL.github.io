"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sum_bitset(elements, k, target):
    """Check if there's a subset of exactly k elements from elements summing to target.
    Uses integer bitsets for the DP."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if target < 0:
        return False
    # Quick bounds check
    srt = sorted(elements)
    if k == n:
        return sum(elements) == target
    if target < sum(srt[:k]) or target > sum(srt[-k:]):
        return False
    # Use complement if cheaper
    ck = n - k
    if ck < k:
        return _subset_sum_bitset(elements, ck, sum(elements) - target)
    # DP: dp[i] is a bitset where bit s is set if s is achievable with i elements
    # Max sum = sum of largest k elements
    max_sum = sum(srt[-k:])
    dp = [0] * (k + 1)
    dp[0] = 1  # bit 0 set: sum 0 achievable with 0 elements
    for e in elements:
        for j in range(min(k, n), 0, -1):
            # Shift dp[j-1] by e, mask to valid range
            shifted = (dp[j-1] << e)
            if max_sum < (1 << 64):
                shifted &= (1 << (max_sum + 1)) - 1
            dp[j] |= shifted
        if (dp[k] >> target) & 1:
            return True
    return False


def _is_valid_set(s):
    """Check if set s satisfies the no-mean constraint."""
    n = len(s)
    slist = sorted(s)
    for i, z in enumerate(slist):
        others = slist[:i] + slist[i+1:]
        for k in range(2, n):
            if _subset_sum_bitset(others, k, k * z):
                return False
    return True


def _can_add(s, c):
    """Check if adding c to set s keeps it valid. s is a list, c is an int not in s."""
    n = len(s)
    if c in s:
        return False
    # Check: c is not the mean of any subset of >=2 elements from s
    if n >= 2:
        for k in range(2, n + 1):
            if _subset_sum_bitset(s, k, k * c):
                return False
    # Check: no element z in s becomes the mean of a subset of >=2 from s∪{c} that includes c
    # z = (sum(others) + c) / (k+1) where others is a subset of s\{z} of size k
    # => sum(others) = (k+1)*z - c
    for z in s:
        others = [x for x in s if x != z]
        m = len(others)
        for k in range(1, m + 1):
            if _subset_sum_bitset(others, k, (k + 1) * z - c):
                return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    # Start with powers of 2 (Sidon set, always valid)
    base = [1 << i for i in range(9)]  # 1,2,4,...,256 -> 9 elements
    assert _is_valid_set(base)

    best = base[:]
    best_sz = len(base)

    # Candidate pool: all integers 1..512 not in base
    cands = [x for x in range(1, 513) if x not in base]

    random.seed(123)

    # Try multiple orderings
    for trial in range(300):
        if time.time() > deadline - 5:
            break

        # Different ordering strategies
        if trial % 3 == 0:
            # Try smallest first
            ordering = sorted(cands)
        elif trial % 3 == 1:
            # Try largest first
            ordering = sorted(cands, reverse=True)
        else:
            # Random
            ordering = cands[:]
            random.shuffle(ordering)

        cur = base[:]
        cur_set = set(base)

        for c in ordering:
            if time.time() > deadline - 3:
                break
            if c in cur_set:
                continue
            if _can_add(cur, c):
                cur.append(c)
                cur_set.add(c)
            if len(cur) >= 48:
                break

        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break

    # If we haven't reached 48, try filling with remaining numbers
    if best_sz < 48:
        best_set = set(best)
        remaining = [x for x in range(1, 513) if x not in best_set]
        random.shuffle(remaining)
        for c in remaining:
            if time.time() > deadline - 1:
                break
            if _can_add(best, c):
                best.append(c)
                best_set.add(c)
                if len(best) >= 48:
                    break

    return best[:48]
# EVOLVE-BLOCK-END
