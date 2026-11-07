import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 84  # 6s safety margin for 90s limit

    def can_add(S_mask, x):
        """Check if S_mask U {x} is a valid set (no forbidden triple)."""
        # Case 1: x is a member of the triple (a = x).
        # Need distinct b, c in S such that x+b, x+c, b+c are all in S U {x}.
        # Since x+b, x+c > x, they must be in S. b+c in S or b+c == x.
        # V = { y in S | x+y in S } = S & (S >> x)
        V_mask = S_mask & (S_mask >> x)
        if V_mask.bit_count() >= 2:
            v_list = []
            temp = V_mask
            while temp:
                lsb = temp & -temp
                v_list.append(lsb.bit_length() - 1)
                temp -= lsb
            n_v = len(v_list)
            for i in range(n_v):
                b = v_list[i]
                for j in range(i + 1, n_v):
                    c = v_list[j]
                    s = b + c
                    if s == x or ((S_mask >> s) & 1):
                        return False

        # Case 2: x is a sum of two members (a+b = x).
        # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
        for a in range(1, x // 2 + 1):
            if ((S_mask >> a) & 1) == 0:
                continue
            b = x - a
            if b == a:
                continue
            if ((S_mask >> b) & 1) == 0:
                continue
            # c in S \ {a,b} such that a+c in S and b+c in S
            U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
            U_mask &= ~(1 << a)
            U_mask &= ~(1 << b)
            if U_mask != 0:
                return False

        return True

    def greedy_construct(order, S_mask):
        """Greedy construction given a fixed order of candidates."""
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, max_passes=30, swap_limit=10):
        """Improve S via additions and systematic swaps (inspiration structure)."""
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb

        not_in_S = [x for x in range(1, 129) if x not in S_set]

        for _ in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            random.shuffle(not_in_S)

            for x in not_in_S[:]:
                if time.time() > deadline:
                    break

                # Try to add x directly
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    S_set.add(x)
                    not_in_S.remove(x)
                    improved = True
                    continue

                # Try swaps: remove y from S, add x
                S_list = list(S_set)
                random.shuffle(S_list)

                found_swap = False
                for y in S_list[:swap_limit]:
                    if time.time() > deadline:
                        break
                    S_temp_mask = S_mask ^ (1 << y)
                    if can_add(S_temp_mask, x):
                        S_mask = S_temp_mask | (1 << x)
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    break

            if not improved:
                break

        return S_mask

    # Precompute a static "conflict degree": for each x, how many other values
    # y it tends to conflict with (i.e., the number of y for which some triple
    # involving x and y is forbidden). Lower degree = less restrictive.
    # We approximate this by counting, for each x, the number of y such that
    # the set {x, y, x+y} or a similar small structure is "dense".
    # A simple proxy: degree[x] = number of y in 1..128 with x+y <= 128
    # (more partners to form sums) -- but we want the RESTRICTIVE ones to be
    # selected LATER, so we sort by degree ascending (least restrictive first).
    degree = [0] * 129
    for x in range(1, 129):
        # Count potential partners y such that x+y <= 128 (x could be a+b=x case)
        # and also count y such that there exists c with x+y, x+c, y+c all <= 128.
        # Simplest: number of y in 1..128 with y != x and x+y <= 128.
        degree[x] = min(128 - x, 128)
    # Lower degree[x] means x is larger (fewer sum-partners) -> less restrictive
    # in the "x is a sum" case, but more restrictive in "x is a member" case.
    # We'll use a blend: prefer middle values. Actually, let's just use the
    # sum-partner count as a heuristic; it's a reasonable proxy.

    # Build degree-ordered list (least restrictive first = smallest degree first)
    degree_order = sorted(range(1, 129), key=lambda x: degree[x])

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)

    best_mask = 0

    # Ascending
    S_mask = greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Descending
    S_mask = greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Degree-ordered (least restrictive first)
    S_mask = greedy_construct(list(degree_order), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Reverse degree-ordered
    S_mask = greedy_construct(list(reversed(degree_order)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Several random orders
    for i in range(40):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Phase 2: Local search from best set
    if time.time() < deadline:
        best_mask = local_search(best_mask, max_passes=30)

    # Phase 3: More random restarts with local search
    for i in range(40):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        S_mask = local_search(S_mask, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)

    # One more local search pass on the final best set
    if time.time() < deadline and best_mask.bit_count() < 128:
        best_mask = local_search(best_mask, max_passes=10)

    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb

    return result
