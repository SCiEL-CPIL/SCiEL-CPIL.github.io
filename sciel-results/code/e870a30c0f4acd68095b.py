import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Start with the safe base: {63, ..., 128}
    # Min pair sum = 63+64 = 127. Possible sums in set: 127, 128.
    # For a triple a,b,c to violate, we need a+b, a+c, b+c all in set.
    # The three sums would be from {127, 128} (with repetition). 
    # If all three are 127: 2(a+b+c)=381, impossible (odd).
    # If two are 127 and one is 128: 2(a+b+c)=382, a+b+c=191.
    #   The sums are 127,127,128. So a+b=127, a+c=127, b+c=128 (or permutations).
    #   From a+b=127 and a+c=127: b=c, contradiction (distinct).
    #   From a+b=127, b+c=128, a+c=127: a=b, contradiction.
    #   All permutations lead to two elements being equal. So safe.
    # If one is 127 and two are 128: 2(a+b+c)=383, impossible (odd).
    # If all three are 128: 2(a+b+c)=384, a+b+c=192.
    #   a+b=128, a+c=128, b+c=128. So a=b=c, contradiction.
    base = set(range(63, N + 1))

    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def compute_summask(mask):
        """Compute bitmask where bit s is set iff there exist two distinct elements in S with sum s."""
        sm = 0
        elems = [x for x in range(1, N+1) if (mask >> x) & 1]
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i+1, n):
                b = elems[j]
                s = a + b
                if s <= N:
                    sm |= (1 << s)
                else:
                    break
        return sm

    def can_add(mask, summask, candidate):
        """Check if adding candidate to S (with membership mask and summask) maintains validity."""
        # Case 1: candidate is a pair-sum in S
        # i.e., there exist a,b in S with a+b == candidate
        # Violation if exists c in S, c!=a, c!=b, with a+c in S and b+c in S
        if (summask >> candidate) & 1:
            # Need to find a,b in S with a+b=candidate, then check for c
            for a in range(1, candidate):
                b = candidate - a
                if b <= a:
                    break
                if not ((mask >> b) & 1):
                    continue
                # a and b are both in S, a+b=candidate
                # Check if there exists c in S, c!=a, c!=b, with a+c in S and b+c in S
                # a+c <= N and b+c <= N
                hi_c = N - max(a, b)
                if hi_c < 1:
                    continue
                for c in range(1, hi_c + 1):
                    if c == a or c == b:
                        continue
                    if not ((mask >> c) & 1):
                        continue
                    ac = a + c
                    bc = b + c
                    if (mask >> ac) & 1 and (mask >> bc) & 1:
                        return False

        # Case 2: candidate is a new member. Check triples (candidate, x, y) with x,y in S
        # Violation if candidate+x, candidate+y, x+y are all in S∪{candidate}
        elems = [x for x in range(1, N+1) if (mask >> x) & 1]
        n = len(elems)
        for i in range(n):
            x = elems[i]
            cx = candidate + x
            if cx > N:
                continue
            cx_in = ((mask >> cx) & 1) or (cx == candidate)
            if not cx_in:
                continue
            for j in range(i + 1, n):
                y = elems[j]
                cy = candidate + y
                if cy > N:
                    continue
                cy_in = ((mask >> cy) & 1) or (cy == candidate)
                if not cy_in:
                    continue
                xy = x + y
                if xy > N:
                    continue
                if (mask >> xy) & 1 or xy == candidate:
                    return False

        return True

    # Greedy: try to add elements from 1 to 62
    S = set(base)
    mask = make_mask(S)
    summask = compute_summask(mask)
    for x in range(1, 63):
        if time.time() > deadline:
            break
        if can_add(mask, summask, x):
            S.add(x)
            mask |= (1 << x)
            summask = compute_summask(mask)

    best_S = set(S)
    best_mask = make_mask(best_S)
    best_summask = compute_summask(best_mask)

    # Try multiple random orderings for greedy construction
    rng = random.Random(42)
    small_elements = list(range(1, 63))
    for trial in range(50000):
        if time.time() > deadline:
            break

        S_try = set(base)
        mask_try = make_mask(S_try)
        summask_try = compute_summask(mask_try)
        rng.shuffle(small_elements)

        for x in small_elements:
            if time.time() > deadline:
                break
            if can_add(mask_try, summask_try, x):
                S_try.add(x)
                mask_try |= (1 << x)
                summask_try = compute_summask(mask_try)

        if len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try
            best_summask = summask_try

    # Local search: try to add any missing element
    for trial in range(50000):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        mask_try = best_mask
        summask_try = best_summask
        improved = False

        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)

        for x in missing:
            if time.time() > deadline:
                break
            if can_add(mask_try, summask_try, x):
                S_try.add(x)
                mask_try |= (1 << x)
                summask_try = compute_summask(mask_try)
                improved = True

        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try
            best_summask = summask_try

    # Swap-based local search: remove one element, try to add one or two others
    for trial in range(20000):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        mask_try = best_mask
        summask_try = best_summask
        improved = False

        # Pick a random element to remove
        elems = list(S_try)
        rng.shuffle(elems)

        for rem in elems:
            if time.time() > deadline:
                break

            S_tmp = set(S_try)
            S_tmp.discard(rem)
            mask_tmp = mask_try & ~(1 << rem)
            summask_tmp = compute_summask(mask_tmp)

            # Try to add elements not in S_tmp
            candidates = [x for x in range(1, N + 1) if x not in S_tmp and x != rem]
            rng.shuffle(candidates)

            added = 0
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(mask_tmp, summask_tmp, x):
                    S_tmp.add(x)
                    mask_tmp |= (1 << x)
                    summask_tmp = compute_summask(mask_tmp)
                    added += 1
                    if added >= 2:
                        break

            # Net gain: added elements - 1 removed
            if added > 1:  # net positive
                if len(S_tmp) > len(best_S):
                    best_S = set(S_tmp)
                    best_mask = mask_tmp
                    best_summask = summask_tmp
                    improved = True
                    break

        if improved:
            break

    # Final polish: try to add any remaining missing elements
    S_try = set(best_S)
    mask_try = best_mask
    summask_try = best_summask
    missing = [x for x in range(1, N + 1) if x not in S_try]
    rng.shuffle(missing)
    for x in missing:
        if time.time() > deadline:
            break
        if can_add(mask_try, summask_try, x):
            S_try.add(x)
            mask_try |= (1 << x)
            summask_try = compute_summask(mask_try)

    if len(S_try) > len(best_S):
        best_S = set(S_try)

    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]
