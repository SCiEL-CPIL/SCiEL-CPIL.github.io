"""Combined: dynamic conflict-degree greedy + incremental counts + remove-to-enable local search."""
import time
import random
from collections import defaultdict

N = 128
DEADLINE = 55.0  # safety margin under 60s (evaluator kills at 90s)

# Precompute forbidden 6-sets: {a, b, c, a+b, a+c, b+c} where a<b<c and b+c <= N
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 <= N:
                _TRIPLES.append((a, b, c, s1, s2, s3))

# Index: for each value, which triple indices it belongs to
_val_to_triples = defaultdict(list)
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if 1 <= v <= N:
            _val_to_triples[v].append(idx)

_NUM_TRIPLES = len(_TRIPLES)


def _build_counts(sel):
    counts = [0] * _NUM_TRIPLES
    for v in sel:
        for ti in _val_to_triples[v]:
            counts[ti] += 1
    return counts


def _is_safe(v, counts):
    for ti in _val_to_triples[v]:
        if counts[ti] == 5:
            return False
    return True


def _greedy_dynamic(remaining, counts, deadline, rng=None):
    """Pick each next value by minimum danger (fewest near-complete triples).
    rng, if given, breaks ties randomly."""
    sel = set()
    result = []
    while remaining and time.time() < deadline:
        best_v = None
        best_danger = None
        # find min-danger value (full scan; 0 is the minimum possible)
        for v in remaining:
            danger = 0
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    danger += 1
            if best_danger is None or danger < best_danger:
                best_danger = danger
                best_v = v
        if best_v is None:
            break
        # commit
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for ti in _val_to_triples[best_v]:
            counts[ti] += 1
    return result


def _local_search(sel, deadline, rng=None):
    """Phase 1: add any safe value. Phase 2: remove one element, add all safe.
    Phase 3: remove two elements, add all safe. Keep if net size increased."""
    counts = _build_counts(sel)
    best_set = set(sel)
    best_size = len(sel)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Phase 1: add safe values
        for v in range(1, N + 1):
            if v in sel:
                continue
            if _is_safe(v, counts):
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True
        if not changed:
            # Phase 2: remove-one-to-enable
            improved = False
            for v_remove in list(sel):
                sel.discard(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] -= 1
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    if _is_safe(v_add, counts):
                        sel.add(v_add)
                        added.append(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] += 1
                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                else:
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
                if time.time() > deadline:
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
                    break
            if improved:
                changed = True
                sel = set(best_set)
                counts = _build_counts(sel)
                continue
            # Phase 3: remove-two-to-enable
            improved = False
            sel_list = list(sel)
            n_sel = len(sel_list)
            for i in range(n_sel):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_sel):
                    if time.time() > deadline:
                        break
                    v1, v2 = sel_list[i], sel_list[j]
                    if v1 in sel and v2 in sel:
                        sel.discard(v1)
                        sel.discard(v2)
                        for ti in _val_to_triples[v1]:
                            counts[ti] -= 1
                        for ti in _val_to_triples[v2]:
                            counts[ti] -= 1
                    else:
                        continue
                    added = []
                    for v_add in range(1, N + 1):
                        if v_add in sel:
                            continue
                        if _is_safe(v_add, counts):
                            sel.add(v_add)
                            added.append(v_add)
                            for ti in _val_to_triples[v_add]:
                                counts[ti] += 1
                    new_size = len(sel)
                    if new_size > best_size:
                        best_set = set(sel)
                        best_size = new_size
                        improved = True
                    else:
                        for v_add in added:
                            sel.discard(v_add)
                            for ti in _val_to_triples[v_add]:
                                counts[ti] -= 1
                        sel.add(v1)
                        for ti in _val_to_triples[v1]:
                            counts[ti] += 1
                        sel.add(v2)
                        for ti in _val_to_triples[v2]:
                            counts[ti] += 1
                    if time.time() > deadline:
                        break
            if improved:
                changed = True
                sel = set(best_set)
                counts = _build_counts(sel)
    return list(best_set)


def _greedy_order(order, deadline):
    """Fixed-order greedy with incremental counts (from current program)."""
    sel = set()
    result = []
    counts = [0] * _NUM_TRIPLES
    for v in order:
        if v in sel:
            continue
        if not _is_safe(v, counts):
            continue
        sel.add(v)
        result.append(v)
        for ti in _val_to_triples[v]:
            counts[ti] += 1
        if time.time() > deadline:
            break
    return result


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = []
    best_size = 0

    def _try(sel):
        nonlocal best, best_size
        sel = _local_search(set(sel), deadline)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # Strategy 1: deterministic dynamic greedy
    remaining = set(range(1, N + 1))
    counts = [0] * _NUM_TRIPLES
    res = _greedy_dynamic(remaining, counts, deadline)
    _try(res)

    # Strategy 2: fixed-order heuristics
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(_val_to_triples[x])),
        sorted(range(1, N + 1), key=lambda x: -len(_val_to_triples[x])),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 2:
            break
        res = _greedy_order(order, deadline)
        _try(res)

    # Strategy 3: randomized restarts (dynamic greedy with random tie-breaks)
    rng = random.Random(12345)
    while time.time() < deadline - 1:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_TRIPLES
        res = _greedy_dynamic(remaining, counts, deadline, rng=rng)
        _try(res)
        if best_size >= 95:
            break
    # Strategy 4: randomized fixed-order restarts
    while time.time() < deadline - 1:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        res = _greedy_order(order, deadline)
        _try(res)
        if best_size >= 95:
            break

    # Final validity check
    if best:
        sel_set = set(best)
        for (a, b, c, s1, s2, s3) in _TRIPLES:
            if (a in sel_set and b in sel_set and c in sel_set
                    and s1 in sel_set and s2 in sel_set and s3 in sel_set):
                best = list(range(N // 2 + 1, N + 1))
                break

    return [int(x) for x in best]
