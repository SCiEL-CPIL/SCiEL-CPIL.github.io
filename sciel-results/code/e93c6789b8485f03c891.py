import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n

def _squarefree_kernel(n):
    f = n
    result = 1
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if f > 1:
        result *= f
    return result

def construct():
    N = 210
    MAX = 10210
    MAX_SIZE = 100

    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit

    # Precompute squarefree kernel for every integer up to MAX (fast).
    sf = [0] * (MAX + 1)
    sf[1] = 1
    for i in range(2, MAX + 1):
        f = i
        res = 1
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    res *= d
                d += 1 if d == 2 else 2
            else:
                d += 1 if d == 2 else 2
        if f > 1:
            res *= f
        sf[i] = res

    target_ker = sf[N]  # squarefree kernel of 210 = 210 (2*3*5*7)

    # We want a subset S of {211..M} such that XOR of sf kernels == target_ker.
    # Two-element solution: exists a,b in (210,M] with sf(a)*sf(b) = target_ker
    #   i.e., sf(a) * target_ker == sf(b) (both squarefree, so product is the XOR).
    # Equivalently sf(b) == sf(a) * target_ker.

    best_two = None  # (M, [a,b])

    # Strategy A: scan M ascending; for each M, check if a two-element pair exists
    # with max == M. Keep the smallest M.
    # We precompute for each kernel value the smallest number >210 that has it.
    # For a two-element pair with max = M: one of a,b is M.
    #   Case 1: b == M. Need a with sf(a) == sf(M) * target_ker.
    #   Case 2: a == M. Need b with sf(b) == sf(M) * target_ker. (same condition)
    # So for each M, let kM = sf(M). We need a number x in (210, M] with sf(x) == kM * target_ker.
    # We can maintain a dict: for each kernel, the list of numbers seen so far (or just existence).

    # Map kernel -> smallest number in (210, current] with that kernel.
    # Actually for two-element we just need existence of some x <= M with sf(x) = kM*target_ker.

    # We'll iterate M from 211 to MAX. Maintain seen_kernels set (kernels of numbers in 211..M).
    # For each M, compute needed = sf(M) * target_ker. If needed in seen_kernels -> two-element pair found with max=M.
    # But we also need the actual x. So store kernel -> one number.

    kernel_to_num = {}  # kernel -> a number in (210, M] with that kernel

    # Also handle the case where x == M itself (i.e., sf(M)*target_ker == sf(M) -> target_ker==1, not our case).
    # So x != M.

    # Iterate
    for M in range(211, MAX + 1):
        if time.time() > deadline:
            break
        kM = sf[M]
        needed = kM * target_ker
        if needed in kernel_to_num:
            x = kernel_to_num[needed]
            # Verify: sf(x) * sf(M) should equal target_ker (as squarefree XOR)
            # Since both squarefree and target_ker squarefree, product == XOR iff disjointness handled.
            # Actually sf(x)*sf(M) as integers: since sf values are squarefree, their product
            # equals the XOR kernel only if they share no prime. Let's just verify directly.
            prod = M * x * N
            if _is_perfect_square(prod):
                best_two = (M, [x, M])
                break
        # Register M's kernel (keep the smallest number for each kernel for compactness,
        # but any works; we want distinct so x != M guaranteed since x in (210,M)).
        if kM not in kernel_to_num:
            kernel_to_num[kM] = M

    if best_two is not None:
        M, pair = best_two
        # pair has 2 distinct elements in (210, M], product*N is square.
        # Validate constraints.
        if 1 <= len(pair) <= MAX_SIZE and all(isinstance(v, int) and not isinstance(v, bool) for v in pair):
            if all(210 < v <= MAX for v in pair) and len(set(pair)) == len(pair):
                if _is_perfect_square(pair[0] * pair[1] * N):
                    return list(pair)

    # If no two-element found within deadline, try three-element search for small M.
    # But first, ensure we have a valid fallback.
    # Fallback: [224, 240] (known to work: 224*240*210 = 2^? ... verified below).
    fallback = [224, 240]
    if _is_perfect_square(fallback[0] * fallback[1] * N):
        return list(fallback)

    # Broader fallback: search small M for any valid 2-element pair (should already be done).
    # If still nothing, do a quick 3-element search.
    # Build kernel -> list of numbers (up to a few) for 211..MAX.
    ker_to_nums = {}
    for i in range(211, MAX + 1):
        k = sf[i]
        if k not in ker_to_nums:
            ker_to_nums[k] = []
        if len(ker_to_nums[k]) < 4:
            ker_to_nums[k].append(i)

    # Three-element: need sf(a)*sf(b)*sf(c) == target_ker (XOR).
    # i.e., sf(a)*sf(b) == target_ker * sf(c) (as squarefree XOR).
    # Iterate c, then for each a with sf(a), need sf(b) == sf(a)*sf(c)*target_ker.
    # This is expensive; do a limited search for minimal max.
    best_three = None
    for c in range(211, MAX + 1):
        if time.time() > deadline:
            break
        kc = sf[c]
        for a in range(211, c):
            if time.time() > deadline:
                break
            ka = sf[a]
            need_b = ka * kc * target_ker
            # need_b must be squarefree (it is, product of squarefree may not be squarefree if overlap)
            # Check existence of b < c (to keep max = c) with sf(b) == need_b, b != a, b != c.
            if need_b in ker_to_nums:
                for b in ker_to_nums[need_b]:
                    if b == a or b == c:
                        continue
                    if b >= c:
                        continue
                    # verify
                    prod = a * b * c * N
                    if _is_perfect_square(prod):
                        M = max(a, b, c)
                        if best_three is None or M < best_three[0]:
                            best_three = (M, [a, b, c])
                        break
        if best_three is not None and best_three[0] <= c:
            # can't get smaller max than c now (since max>=c and we found one with max<=c)
            break

    if best_three is not None:
        M, triple = best_three
        if _is_perfect_square(triple[0] * triple[1] * triple[2] * N):
            return list(triple)

    return list(fallback)
