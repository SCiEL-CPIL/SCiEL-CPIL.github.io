"""Rewritten program: structural core (1 + powers of 2) + greedy + local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))

    # Precompute full LCM table for O(1) lookups
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        li = lcm_table[i]
        for j in range(1, n + 1):
            li[j] = _lcm(i, j)

    def get_lcm(a, b):
        return lcm_table[a][b]

    def is_bad_triple(a, b, x):
        lab = get_lcm(a, b)
        if lab != get_lcm(a, x):
            return False
        return lab == get_lcm(b, x)

    # ---------- Structural core: 1 + powers of 2 ----------
    # 1 is in no bad triple (lcm(1,x)=x). Powers of 2 form a chain:
    # for 2^i < 2^j, lcm(2^i,2^j)=2^j, and 2^j's other LCMs differ -> safe.
    core = [1]
    p = 2
    while p <= n:
        core.append(p)
        p *= 2
    core_set = set(core)
    # Verify core is valid (defensive)
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            for k in range(j + 1, len(core)):
                if is_bad_triple(core[i], core[j], core[k]):
                    core = [1]
                    core_set = {1}
                    break

    # ---------- Greedy add a candidate to a given set ----------
    def can_add(x, sel_list, sel_set):
        if x in sel_set:
            return False
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i + 1, len(sel_list)):
                b = sel_list[j]
                if is_bad_triple(a, b, x):
                    return False
        return True

    def greedy_fill(start_list, start_set, order):
        sel_list = list(start_list)
        sel_set = set(start_set)
        for x in order:
            if x in sel_set:
                continue
            if can_add(x, sel_list, sel_set):
                sel_list.append(x)
                sel_set.add(x)
        return sel_list

    # ---------- Orderings for greedy ----------
    def order_by_popcount_omega():
        # Prefer numbers with few prime factors (more "rigid", less likely to
        # complete a triple), ties broken by value.
        def omega(x):
            c = 0
            d = x
            pf = 2
            while pf * pf <= d:
                if d % pf == 0:
                    c += 1
                    while d % pf == 0:
                        d //= pf
                pf += 1
            if d > 1:
                c += 1
            return c
        return sorted(universe, key=lambda x: (omega(x), x))

    def order_descending():
        return sorted(universe, reverse=True)

    def order_by_lcm_spread():
        # Prefer numbers whose LCMs with others are large/distinct (less equal-lcm risk)
        return sorted(universe, key=lambda x: -max(get_lcm(x, y) for y in universe))

    def order_random(seed):
        random.seed(seed)
        lst = list(universe)
        random.shuffle(lst)
        return lst

    # ---------- Local search: 1-remove + best-add swaps ----------
    def local_search(initial_list, max_iters=400):
        sel_list = list(initial_list)
        sel_set = set(sel_list)
        best = len(sel_list)
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try each removal, then best valid addition
            for ri in range(len(sel_list)):
                rem = sel_list[ri]
                rem_list = sel_list[:ri] + sel_list[ri + 1:]
                rem_set = sel_set - {rem}
                # find a valid add not in rem_set
                best_add = None
                # prefer adding something that keeps size up; scan for first valid
                for x in universe:
                    if x in rem_set:
                        continue
                    if can_add(x, rem_list, rem_set):
                        best_add = x
                        break
                if best_add is not None and best_add != rem:
                    new_list = rem_list + [best_add]
                    new_set = rem_set | {best_add}
                    if len(new_list) >= len(sel_list):
                        sel_list = new_list
                        sel_set = new_set
                        best = len(sel_list)
                        improved = True
                        break
            if not improved:
                break
        return sel_list

    # ---------- Run search ----------
    best_result = list(core)

    # Deterministic greedy fills with different orderings
    for order_fn in (order_by_popcount_omega, order_descending, order_by_lcm_spread):
        if time.time() - start_time > DEADLINE:
            break
        res = greedy_fill(core, core_set, order_fn())
        if len(res) > len(best_result):
            best_result = res

    # Local search on the best so far
    if time.time() - start_time < DEADLINE - 3:
        imp = local_search(best_result, max_iters=300)
        if len(imp) > len(best_result):
            best_result = imp

    # Randomized restarts: greedy with random order + local search
    attempt = 0
    while time.time() - start_time < DEADLINE - 3:
        attempt += 1
        order = order_random(1000 + attempt * 2654435761)
        res = greedy_fill(core, core_set, order)
        if len(res) > len(best_result):
            best_result = res
        if time.time() - start_time < DEADLINE - 3:
            imp = local_search(res, max_iters=150)
            if len(imp) > len(best_result):
                best_result = imp

    # Also try pure local search from a few random greedy starts without core bias
    for s in range(3):
        if time.time() - start_time < DEADLINE - 3:
            order = order_random(900000 + s * 1234567)
            res = greedy_fill([], set(), order)
            imp = local_search(res, max_iters=200)
            if len(imp) > len(best_result):
                best_result = imp

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
