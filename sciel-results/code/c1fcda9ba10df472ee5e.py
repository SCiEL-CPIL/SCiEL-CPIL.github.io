"""Hypergraph independent set for the 6-set forbidden constraint.
Design family: precompute forbidden 6-sets, incremental counts, dynamic-conflict
greedy + clause-weighted greedy + remove-to-enable local search + 2-opt swap polish.
"""
import time
import random
from collections import defaultdict

N = 128
DEADLINE = 82.0  # safety margin under 90s wall-clock kill

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

T = len(_TRIPLES)

# Index: for each value, which triples it belongs to
_val_to_triples = defaultdict(list)
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if 1 <= v <= N:
            _val_to_triples[v].append(idx)

# For each value, the list of triples (as indices) it appears in — precomputed
_val_triples = [None] + [list(_val_to_triples[v]) for v in range(1, N + 1)]


def _is_valid(sel_set):
    for (a, b, c, s1, s2, s3) in _TRIPLES:
        if a in sel_set and b in sel_set and c in sel_set and s1 in sel_set and s2 in sel_set and s3 in sel_set:
            return False
    return True


def _greedy_dynamic(deadline, rng=None, seed_bias=None):
    """Dynamic greedy: pick the unselected value with fewest near-complete (count==5) triples.
    Uses incremental danger tracking for O(1) per-value danger query."""
    sel = set()
    result = []
    counts = [0] * T
    # danger[v] = number of triples containing v that currently have count==5
    danger = [0] * (N + 1)
    remaining = list(range(1, N + 1))
    rem_set = set(remaining)

    # If seed_bias is provided, use it to break ties or bias initial selection
    if seed_bias is not None:
        # Pre-sort by bias to influence tie-breaking
        remaining.sort(key=lambda v: seed_bias.get(v, 0))

    while remaining and time.time() < deadline:
        # Find value with minimum danger among remaining
        best_v = None
        best_d = None
        for v in remaining:
            d = danger[v]
            if best_d is None or d < best_d:
                best_d = d
                best_v = v
            if d == 0:
                break

        if best_v is None:
            break

        if best_d > 0:
            # No zero-danger value; check if any is truly safe (shouldn't happen if danger is accurate)
            # Fallback: just break
            break

        # Commit best_v
        sel.add(best_v)
        result.append(best_v)
        rem_set.discard(best_v)
        remaining.remove(best_v)

        # Update counts and danger
        for ti in _val_triples[best_v]:
            old_count = counts[ti]
            counts[ti] = old_count + 1
            if old_count == 4:
                # This triple just became count==5; increment danger for all other 5 members
                for v2 in _TRIPLES[ti]:
                    if v2 != best_v and v2 in rem_set:
                        danger[v2] += 1

    return result


def _greedy_weighted(deadline, rng=None):
    """Clause-weighted greedy: at each step, pick the value that minimizes the
    weighted sum of 'near-completion' across its triples. Weight of a triple at
    count k is (6-k)^2, giving a smooth surrogate for the hard constraint."""
    sel = set()
    result = []
    counts = [0] * T
    remaining = list(range(1, N + 1))
    rem_set = set(remaining)

    while remaining and time.time() < deadline:
        best_v = None
        best_score = None

        for v in remaining:
            score = 0
            for ti in _val_triples[v]:
                k = counts[ti]
                if k >= 5:
                    score += 1000  # hard block
                    break
                # Smooth surrogate: (6-k)^2 gives increasing penalty as triple fills
                score += (6 - k) ** 2
            if best_score is None or score < best_score:
                best_score = score
                best_v = v
            if score <= 36:  # (6-0)^2 = 36, can't do much better
                # Early exit if we find a very low-score candidate
                pass

        if best_v is None:
            break

        if best_score >= 1000:
            break  # No safe value

        sel.add(best_v)
        result.append(best_v)
        rem_set.discard(best_v)
        remaining.remove(best_v)

        for ti in _val_triples[best_v]:
            counts[ti] += 1

    return result


def _local_search(sel_list, deadline, rng=None):
    """Remove-to-enable local search: try adding any safe value; if stuck, try
    removing one element and re-adding greedily to see if net size improves."""
    if not sel_list:
        return []

    sel = set(sel_list)
    counts = [0] * T
    for v in sel:
        for ti in _val_triples[v]:
            counts[ti] += 1

    best_set = set(sel)
    best_size = len(sel)

    changed = True
    while changed and time.time() < deadline:
        changed = False

        # Phase 1: Add any safe value
        for v in range(1, N + 1):
            if v in sel:
                continue
            safe = True
            for ti in _val_triples[v]:
                if counts[ti] == 5:
                    safe = False
                    break
            if safe:
                sel.add(v)
                for ti in _val_triples[v]:
                    counts[ti] += 1
                changed = True

        if not changed:
            # Phase 2: Try removing one element to enable more additions
            improved = False
            for v_remove in list(sel):
                if time.time() > deadline:
                    break

                # Tentatively remove v_remove
                sel.discard(v_remove)
                for ti in _val_triples[v_remove]:
                    counts[ti] -= 1

                # Try to add as many safe values as possible
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    safe = True
                    for ti in _val_triples[v_add]:
                        if counts[ti] == 5:
                            safe = False
                            break
                    if safe:
                        sel.add(v_add)
                        added.append(v_add)
                        for ti in _val_triples[v_add]:
                            counts[ti] += 1

                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                else:
                    # Revert
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_triples[v_remove]:
                        counts[ti] += 1

            if improved:
                changed = True
                # Reset to best
                sel = set(best_set)
                counts = [0] * T
                for v in sel:
                    for ti in _val_triples[v]:
                        counts[ti] += 1

    return list(best_set)


def _swap_polish(sel_list, deadline, rng=None):
    """2-opt swap polish: try swapping one selected element with one unselected
    element if the swap enables a net size increase."""
    if not sel_list:
        return []

    sel = set(sel_list)
    counts = [0] * T
    for v in sel:
        for ti in _val_triples[v]:
            counts[ti] += 1

    best_set = set(sel)
    best_size = len(sel)
    improved = True

    while improved and time.time() < deadline:
        improved = False
        for v_in in list(sel):
            if time.time() > deadline:
                break
            # Remove v_in
            sel.discard(v_in)
            for ti in _val_triples[v_in]:
                counts[ti] -= 1

            # Try each unselected v_out
            for v_out in range(1, N + 1):
                if v_out in sel:
                    continue
                # Check if v_out is safe to add
                safe = True
                for ti in _val_triples[v_out]:
                    if counts[ti] == 5:
                        safe = False
                        break
                if not safe:
                    continue

                # Add v_out
                sel.add(v_out)
                for ti in _val_triples[v_out]:
                    counts[ti] += 1

                # Now try to add any other safe values
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    safe2 = True
                    for ti in _val_triples[v_add]:
                        if counts[ti] == 5:
                            safe2 = False
                            break
                    if safe2:
                        sel.add(v_add)
                        added.append(v_add)
                        for ti in _val_triples[v_add]:
                            counts[ti] += 1

                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                    # Reset to best and continue outer loop
                    sel = set(best_set)
                    counts = [0] * T
                    for v in sel:
                        for ti in _val_triples[v]:
                            counts[ti] += 1
                    break
                else:
                    # Revert: remove added, remove v_out, add back v_in
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_triples[v_add]:
                            counts[ti] -= 1
                    sel.discard(v_out)
                    for ti in _val_triples[v_out]:
                        counts[ti] -= 1
                    sel.add(v_in)
                    for ti in _val_triples[v_in]:
                        counts[ti] += 1

    return list(best_set)


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = []
    best_size = 0

    def _try_update(res):
        nonlocal best, best_size
        if res and len(res) > best_size:
            # Validate
            s = set(res)
            if _is_valid(s):
                best = list(res)
                best_size = len(res)

    # Strategy 1: Deterministic dynamic greedy
    res = _greedy_dynamic(deadline)
    res = _local_search(res, deadline)
    res = _swap_polish(res, deadline)
    _try_update(res)

    # Strategy 2: Weighted greedy
    res = _greedy_weighted(deadline)
    res = _local_search(res, deadline)
    res = _swap_polish(res, deadline)
    _try_update(res)

    # Strategy 3: Randomized restarts with dynamic greedy
    rng = random.Random(12345)
    while time.time() < deadline - 3:
        # Random bias for tie-breaking
        bias = {}
        for v in range(1, N + 1):
            bias[v] = rng.random()
        res = _greedy_dynamic(deadline, rng=rng, seed_bias=bias)
        res = _local_search(res, deadline, rng=rng)
        res = _swap_polish(res, deadline, rng=rng)
        _try_update(res)

        if best_size >= 95:
            break

    # Final validity check and fallback
    if best:
        if not _is_valid(set(best)):
            # Fallback: upper half is always valid (all values > 64, so a+b > 64+65 > 128 for any a,b in set... wait)
            # Actually upper half 65..128: a+b for a,b in 65..128 can be up to 256, but we need a+b <= 128.
            # If a,b,c all in 65..128, then a+b >= 65+66=131 > 128, so no triple can have all 6 in set.
            best = list(range(65, N + 1))
            best_size = len(best)
    else:
        best = list(range(65, N + 1))
        best_size = len(best)

    return [int(x) for x in best]
