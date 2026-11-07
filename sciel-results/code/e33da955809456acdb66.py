import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Start with the safe base: {65, ..., 128}
    base = set(range(65, N + 1))

    # Bitmask for O(1) membership tests
    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def can_add(mask, S, candidate):
        """Check if adding candidate to S (with membership mask) maintains validity."""
        elems = list(S)
        n = len(elems)
        # 1. Check triples where candidate is a member: (candidate, x, y)
        #    Violation if (candidate+x), (candidate+y), (x+y) are all in S∪{candidate}
        for i in range(n):
            x = elems[i]
            cx = candidate + x
            if cx > N:
                continue
            # cx in S_new? (cx in S, or cx == candidate which is trivially true since candidate is being added)
            if not (mask >> cx) & 1 and cx != candidate:
                continue
            for j in range(i + 1, n):
                y = elems[j]
                cy = candidate + y
                if cy > N:
                    continue
                if not (mask >> cy) & 1 and cy != candidate:
                    continue
                xy = x + y
                if xy > N:
                    continue
                if (mask >> xy) & 1 or xy == candidate:
                    return False

        # 2. Check triples where candidate is a sum: (a, b, c) with a+b == candidate
        #    Violation if a+c and b+c are in S
        for i in range(n):
            a = elems[i]
            if a >= candidate:
                break
            b = candidate - a
            if b <= a:
                continue
            if not (mask >> b) & 1:
                continue
            # Check if there is a c in S (c != a, c != b) such that a+c in S and b+c in S
            for c in elems:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    continue
                if (mask >> ac) & 1 and (mask >> bc) & 1:
                    return False

        return True

    # Greedy: try to add elements from 1 to 64
    S = set(base)
    mask = make_mask(S)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if can_add(mask, S, x):
            S.add(x)
            mask |= (1 << x)

    best_S = set(S)
    best_mask = make_mask(best_S)

    # Try multiple random orderings for greedy construction
    rng = random.Random(42)
    greedy_trials = 0
    while time.time() < deadline - 10:
        greedy_trials += 1
        S_try = set(base)
        mask_try = make_mask(S_try)
        small_elements = list(range(1, 65))
        rng.shuffle(small_elements)

        for x in small_elements:
            if time.time() > deadline - 10:
                break
            if can_add(mask_try, S_try, x):
                S_try.add(x)
                mask_try |= (1 << x)

        if len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try

    # Local search: try to add any missing element
    add_trials = 0
    while time.time() < deadline - 5:
        add_trials += 1
        S_try = set(best_S)
        mask_try = best_mask
        improved = False

        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)

        for x in missing:
            if time.time() > deadline - 5:
                break
            if can_add(mask_try, S_try, x):
                S_try.add(x)
                mask_try |= (1 << x)
                improved = True

        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try

    # Swap-based local search: remove one element, try to add one or two others
    swap_trials = 0
    while time.time() < deadline - 2:
        swap_trials += 1
        S_try = set(best_S)
        mask_try = best_mask
        improved = False

        # Pick a random element to remove
        elems = list(S_try)
        rng.shuffle(elems)

        for rem in elems:
            if time.time() > deadline - 2:
                break

            S_tmp = set(S_try)
            S_tmp.discard(rem)
            mask_tmp = mask_try & ~(1 << rem)

            # Try to add elements not in S_tmp
            candidates = [x for x in range(1, N + 1) if x not in S_tmp and x != rem]
            rng.shuffle(candidates)

            added = 0
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                if can_add(mask_tmp, S_tmp, x):
                    S_tmp.add(x)
                    mask_tmp |= (1 << x)
                    added += 1
                    if added >= 2:
                        break

            # Net gain: added elements - 1 removed
            if added > 1:  # net positive
                if len(S_tmp) > len(best_S):
                    best_S = set(S_tmp)
                    best_mask = mask_tmp
                    improved = True
                    break
            elif added == 1 and len(S_tmp) > len(best_S):
                # Same size or better, but we want strictly larger
                pass

        if improved:
            break

    # Remove-2-add-3 local search: remove two elements, try to add three others (net +1)
    r2a3_trials = 0
    while time.time() < deadline - 1:
        r2a3_trials += 1
        S_try = set(best_S)
        mask_try = best_mask
        improved = False

        elems = list(S_try)
        rng.shuffle(elems)
        n_elems = len(elems)

        # Try pairs of elements to remove
        for i in range(min(n_elems, 15)):
            if time.time() > deadline - 1:
                break
            rem1 = elems[i]
            for j in range(i + 1, min(n_elems, 15)):
                if time.time() > deadline - 1:
                    break
                rem2 = elems[j]

                S_tmp = set(S_try)
                S_tmp.discard(rem1)
                S_tmp.discard(rem2)
                mask_tmp = mask_try & ~(1 << rem1) & ~(1 << rem2)

                # Try to add elements not in S_tmp
                candidates = [x for x in range(1, N + 1) if x not in S_tmp and x != rem1 and x != rem2]
                rng.shuffle(candidates)

                added = 0
                for x in candidates:
                    if time.time() > deadline - 1:
                        break
                    if can_add(mask_tmp, S_tmp, x):
                        S_tmp.add(x)
                        mask_tmp |= (1 << x)
                        added += 1
                        if added >= 3:
                            break

                # Net gain: added - 2 removed; need added >= 3 for net +1
                if added >= 3 and len(S_tmp) > len(best_S):
                    best_S = set(S_tmp)
                    best_mask = mask_tmp
                    improved = True
                    break
            if improved:
                break

        if improved:
            break

    # Final polish: try to add any remaining missing elements
    S_try = set(best_S)
    mask_try = best_mask
    missing = [x for x in range(1, N + 1) if x not in S_try]
    rng.shuffle(missing)
    for x in missing:
        if time.time() > deadline:
            break
        if can_add(mask_try, S_try, x):
            S_try.add(x)
            mask_try |= (1 << x)

    if len(S_try) > len(best_S):
        best_S = set(S_try)

    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]
