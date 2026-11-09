import time
import random
from itertools import combinations

N = 192
MAX_K = 3
MAX_SIZE = 48

# Precompute: for each a in 1..192, the set of target sums S (<=3*192=576)
# such that a divides S. We need to check sums of 3 distinct OTHER elements.
# For a given a, we need to know if any 3-sum from others is a multiple of a.

def _precompute_divs():
    """divs[a] = set of multiples of a up to 3*192"""
    divs = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for m in range(a, 3 * N + 1, a):
            divs[a].add(m)
    return divs

DIVS = _precompute_divs()

def _is_valid(selected):
    """Check validity: no element divides sum of any 3 distinct others."""
    m = len(selected)
    if m < 4:
        return True
    for idx, a in enumerate(selected):
        others = [selected[j] for j in range(m) if j != idx]
        # sums of 3 distinct others
        for i in range(len(others)):
            oi = others[i]
            for j in range(i + 1, len(others)):
                oj = others[j]
                for k in range(j + 1, len(others)):
                    s = oi + oj + others[k]
                    if s % a == 0:
                        return False
    return True

def _can_add(x, selected):
    """Check if x can be added to selected (list) without violating constraints."""
    m = len(selected)
    # 1) x divides sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            a = selected[i]
            for j in range(i + 1, m):
                b = selected[j]
                for k in range(j + 1, m):
                    s = a + b + selected[k]
                    if s % x == 0:
                        return False
    # 2) for each a in selected, a divides sum of 3 from (selected ∪ {x}) \ {a}
    #    i.e., a divides (a1+a2+x) where a1,a2 are two distinct others (not a)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(m) if j != i]
        if len(others) >= 2:
            for p in range(len(others)):
                op = others[p]
                for q in range(p + 1, len(others)):
                    s = op + others[q] + x
                    if s % a == 0:
                        return False
    return True

def _can_add_fast(x, selected):
    """Faster: use precomputed divisibility via sum membership."""
    m = len(selected)
    if m >= 3:
        for i in range(m):
            a = selected[i]
            for j in range(i + 1, m):
                b = selected[j]
                for k in range(j + 1, m):
                    s = a + b + selected[k]
                    if s in DIVS[x]:
                        return False
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(m) if j != i]
        if len(others) >= 2:
            for p in range(len(others)):
                op = others[p]
                for q in range(p + 1, len(others)):
                    s = op + others[q] + x
                    if s in DIVS[a]:
                        return False
    return True

def _greedy_build(order, limit=MAX_SIZE):
    selected = []
    for x in order:
        if len(selected) >= limit:
            break
        if _can_add_fast(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    start = time.time()
    best = list(selected)
    best_size = len(best)

    def try_add_any(cur):
        cands = [x for x in range(1, N + 1) if x not in cur]
        for x in cands:
            if _can_add_fast(x, cur):
                return x
        return None

    while time.time() - start < time_limit:
        improved = False
        # 1-add
        x = try_add_any(best)
        if x is not None:
            best.append(x)
            improved = True
        else:
            # 1-swap
            done = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    done = True
                    break
                temp = best[:i] + best[i + 1:]
                x = try_add_any(temp)
                if x is not None:
                    best = temp + [x]
                    improved = True
                    done = True
                    break
            if done:
                break
            if improved:
                continue
            # 2-swap
            n = len(best)
            cands = [x for x in range(1, N + 1) if x not in best]
            found = False
            for i in range(n):
                if time.time() - start > time_limit:
                    found = True
                    break
                for j in range(i + 1, n):
                    if time.time() - start > time_limit:
                        found = True
                        break
                    temp = best[:i] + best[i + 1:j] + best[j + 1:]
                    for a in cands:
                        if time.time() - start > time_limit:
                            found = True
                            break
                        if not _can_add_fast(a, temp):
                            continue
                        for b in cands:
                            if b == a:
                                continue
                            if _can_add_fast(b, temp + [a]):
                                best = temp + [a, b]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
            if improved:
                continue
            # remove-one-add-two
            if len(best) > 8:
                found = False
                cands = [x for x in range(1, N + 1) if x not in best]
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        found = True
                        break
                    temp = best[:i] + best[i + 1:]
                    for a in cands:
                        if time.time() - start > time_limit:
                            found = True
                            break
                        if not _can_add_fast(a, temp):
                            continue
                        for b in cands:
                            if b == a:
                                continue
                            if _can_add_fast(b, temp + [a]):
                                best = temp + [a, b]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not improved:
                break
        if len(best) > best_size:
            best_size = len(best)
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    best_result = []

    # Greedy large-first
    r = _greedy_build(list(range(N, 0, -1)))
    best_result = r

    # Greedy small-first
    r = _greedy_build(list(range(1, N + 1)))
    if len(r) > len(best_result):
        best_result = r

    # Randomized greedy
    for seed in range(200):
        if time.time() > deadline - 25:
            break
        random.seed(seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r

    # Interval-based: prefer mid-range (avoid very small divisors and very large)
    for lo, hi in [(60, 192), (40, 192), (80, 192), (50, 150), (70, 170)]:
        if time.time() > deadline - 25:
            break
        order = list(range(hi, lo - 1, -1))
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r

    # Local search on best
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        best_result = _local_search(best_result, remaining)

    # Local search on other good starts
    for seed in range(60):
        if time.time() > deadline - 5:
            break
        random.seed(5000 + seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) >= len(best_result) - 3:
            remaining = min(deadline - time.time() - 2, 4.0)
            if remaining > 0.5:
                r = _local_search(r, remaining)
                if len(r) > len(best_result):
                    best_result = r

    # Final polish on best
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 0.3:
            best_result = _local_search(best_result, remaining)

    return best_result
