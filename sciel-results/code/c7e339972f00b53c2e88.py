"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]

    # LCM for integer arithmetic
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
    weights = [L // d for d in denominators]

    # ---------- Meet-in-the-middle exact search ----------
    def mitm_exact():
        mid = n // 2  # 23
        w1 = weights[:mid]
        w2 = weights[mid:]
        n1 = len(w1)
        n2 = len(w2)

        def gen_sums(wlist, timeout_check):
            """Generate all subset sums < L with their masks."""
            sums = [(0, 0)]
            for i, w in enumerate(wlist):
                if timeout_check():
                    return None
                bit = 1 << i
                new_sums = []
                for s, m in sums:
                    ns = s + w
                    if ns < L:
                        new_sums.append((ns, m | bit))
                sums.extend(new_sums)
            return sums

        def check_time():
            return time.time() > deadline - 40.0

        # Generate sums for first half
        sums1 = gen_sums(w1, check_time)
        if sums1 is None:
            return None

        # Generate sums for second half
        sums2 = gen_sums(w2, check_time)
        if sums2 is None:
            return None

        # Sort sums2 by value for binary search
        sums2.sort(key=lambda x: x[0])
        vals2 = [s for s, _ in sums2]

        best_sum = -1
        best_mask = 0

        for s1, m1 in sums1:
            if check_time():
                return None
            # Find largest s2 such that s1 + s2 < L
            target = L - s1 - 1
            idx = bisect.bisect_right(vals2, target)
            if idx > 0:
                s2, m2 = sums2[idx - 1]
                total = s1 + s2
                if total > best_sum:
                    best_sum = total
                    best_mask = m1 | (m2 << n1)

        if best_sum < 0:
            return None

        result = []
        for i in range(n):
            if best_mask & (1 << i):
                result.append(i)
        return result, best_sum, L

    # ---------- Greedy initialization ----------
    def greedy(order):
        selected = set()
        total = 0
        for idx in order:
            w = weights[idx]
            if total + w < L:
                selected.add(idx)
                total += w
        return selected, total

    # ---------- Local search refinement ----------
    def local_search(start_set, max_iterations=50000):
        cur = set(start_set)
        if not cur:
            return cur, 0
        cur_sum = sum(weights[i] for i in cur)
        if cur_sum >= L:
            return cur, cur_sum

        best_sum = cur_sum
        best_set = set(cur)
        not_in_set = set(range(n)) - cur

        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break

            improved = False
            # Try adding one element
            if not_in_set:
                candidates = list(not_in_set)
                random.shuffle(candidates)
                for add in candidates:
                    ns = cur_sum + weights[add]
                    if ns < L and ns > best_sum:
                        best_sum = ns
                        best_set = cur | {add}
                        improved = True
                        break

            # Try removing one and adding another
            if not improved and len(cur) > 0 and not_in_set:
                cur_list = list(cur)
                not_in_list = list(not_in_set)
                random.shuffle(cur_list)
                random.shuffle(not_in_list)
                for rm in cur_list[:20]:
                    for add in not_in_list[:20]:
                        ns = cur_sum - weights[rm] + weights[add]
                        if ns < L and ns > best_sum:
                            best_sum = ns
                            best_set = (cur - {rm}) | {add}
                            improved = True
                            break
                    if improved:
                        break

            if improved:
                cur = set(best_set)
                cur_sum = best_sum
                not_in_set = set(range(n)) - cur
            else:
                break

        return cur, best_sum

    # ---------- Main flow ----------
    best_indices = []
    best_sum = 0

    # Phase 1: Greedy baselines
    random.seed(42)
    orderings = [list(range(n)), list(reversed(range(n)))]
    for _ in range(30):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs and total > best_sum:
            best_sum = total
            best_indices = sorted(idxs)

    # Phase 2: Meet-in-the-middle exact search
    if time.time() < deadline - 45.0:
        result = mitm_exact()
        if result is not None:
            mitm_indices, mitm_sum, _ = result
            if mitm_indices and mitm_sum > best_sum:
                best_sum = mitm_sum
                best_indices = sorted(mitm_indices)

    # Phase 3: Local search refinement
    if best_indices and time.time() < deadline - 15.0:
        refined_set, refined_sum = local_search(set(best_indices))
        if refined_sum > best_sum:
            best_sum = refined_sum
            best_indices = sorted(refined_set)

    # Verify with exact fractions
    if best_indices:
        total_frac = sum(unit_vals[i] for i in best_indices)
        if total_frac >= 1:
            # Remove elements until valid
            best_indices.sort(key=lambda i: -unit_vals[i])
            while best_indices and sum(unit_vals[i] for i in best_indices) >= 1:
                best_indices.pop()
            if not best_indices:
                best_indices = [0]

    if not best_indices:
        best_indices = [0]

    return sorted(best_indices)
# EVOLVE-BLOCK-END
