"""Minimum maximal no-isosceles set on N x N grid.

Structural redesign:
  * Precompute the full isosceles-triple table once.
  * Represent "can add point p" as a bitmask intersection (O(1)).
  * Fast re-maximization, local search (remove + pairwise swap), and a
    conflict-graph greedy seed.
"""
import time
import random
import itertools
import numpy as np

N = 16
NUM = N * N


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2_ = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2_ or d1 == d3 or d2_ == d3


def _precompute():
    """Return (conflict, iso_pairs).
    conflict[p] = bitmask of q != p such that p,q cannot both be in a
                  no-isosceles set (i.e. some r makes an iso triangle).
    iso_pairs[r] = set of (i,j) with i<j, i,j != r, that form an iso
                  triangle with r.
    """
    pts = list(itertools.product(range(N), repeat=2))
    idx = {p: i for i, p in enumerate(pts)}
    conflict = [0] * NUM
    iso_pairs = [set() for _ in range(NUM)]

    # Precompute all squared distances (NUM x NUM)
    dmat = np.empty((NUM, NUM), dtype=np.int32)
    xs = np.array([p[0] for p in pts], dtype=np.int32)
    ys = np.array([p[1] for p in pts], dtype=np.int32)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dmat = dx * dx + dy * dy

    for r in range(NUM):
        # group other points by squared distance to r
        by_dist = {}
        for q in range(NUM):
            if q == r:
                continue
            dmat_r_q = int(dmat[r, q])
            by_dist.setdefault(dmat_r_q, []).append(q)
        # for each distance group of size >= 2, all pairs conflict
        for grp in by_dist.values():
            m = len(grp)
            if m >= 2:
                for a in range(m):
                    ia = grp[a]
                    for b in range(a + 1, m):
                        ib = grp[b]
                        if ia < ib:
                            iso_pairs[r].add((ia, ib))
                        else:
                            iso_pairs[r].add((ib, ia))
        # build conflict bitmask for r from iso_pairs[r]
        mask = 0
        for (i, j) in iso_pairs[r]:
            mask |= (1 << i) | (1 << j)
        conflict[r] = mask

    return conflict, iso_pairs


def _can_add_bit(p, chosen_mask, conflict):
    """True if point p can be added to the set represented by chosen_mask."""
    if (chosen_mask >> p) & 1:
        return False
    # p conflicts with any chosen point?
    return (conflict[p] & chosen_mask) == 0


def _maximalize(chosen_mask, all_indices, conflict):
    """Add points (in all_indices order) until no more can be added."""
    cm = chosen_mask
    for p in all_indices:
        if (cm >> p) & 1:
            continue
        if (conflict[p] & cm) == 0:
            cm |= (1 << p)
    return cm


def _is_maximal(chosen_mask, all_indices, conflict):
    for p in all_indices:
        if (chosen_mask >> p) & 1:
            continue
        if (conflict[p] & chosen_mask) == 0:
            return False
    return True


def _is_valid_no_iso(chosen, iso_pairs):
    """Verify no three points form an isosceles triangle."""
    if len(chosen) < 3:
        return True
    idx_set = set(chosen)
    for r in idx_set:
        for (i, j) in iso_pairs[r]:
            if i in idx_set and j in idx_set:
                return False
    return True


def _greedy_conflict_order(all_indices, conflict):
    """Greedy: repeatedly pick the point with fewest conflicts to the set."""
    cm = 0
    remaining = set(all_indices)
    # initial ordering by total conflict degree (fewest first)
    order = sorted(all_indices, key=lambda p: bin(conflict[p]).count('1'))
    # iterate; recompute is cheap enough
    for p in order:
        if p not in remaining:
            continue
        if (conflict[p] & cm) == 0:
            cm |= (1 << p)
            remaining.discard(p)
    return cm


def _greedy_order(order, conflict):
    cm = 0
    for p in order:
        if (conflict[p] & cm) == 0:
            cm |= (1 << p)
    return cm


def _local_search(chosen_mask, all_indices, conflict, iso_pairs,
                  max_iters=200000, deadline=None, seed=0):
    """Local search with remove and pairwise-swap moves."""
    current = chosen_mask
    best = chosen_mask
    best_len = bin(current).count('1')
    rng = random.Random(seed)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if current == 0:
            break
        improved = False

        # --- try remove moves (in random order) ---
        chosen_list = [p for p in all_indices if (current >> p) & 1]
        rng.shuffle(chosen_list)
        for p in chosen_list:
            if deadline is not None and time.time() > deadline:
                break
            new_mask = current & ~(1 << p)
            new_mask = _maximalize(new_mask, all_indices, conflict)
            new_len = bin(new_mask).count('1')
            if new_len < best_len:
                current = new_mask
                best = new_mask
                best_len = new_len
                improved = True
                break
            if new_len < bin(current).count('1'):
                current = new_mask
                improved = True
                break

        if improved:
            continue

        # --- try pairwise swap moves ---
        chosen_list = [p for p in all_indices if (current >> p) & 1]
        rng.shuffle(chosen_list)
        # limit swap attempts for speed
        swap_attempts = 0
        for i in range(len(chosen_list)):
            if deadline is not None and time.time() > deadline:
                break
            if swap_attempts > 40:
                break
            pi = chosen_list[i]
            for j in range(i + 1, len(chosen_list)):
                if swap_attempts > 40:
                    break
                pj = chosen_list[j]
                swap_attempts += 1
                # remove both, add back
                base = current & ~(1 << pi) & ~(1 << pj)
                # try adding just pi
                cand1 = base | (1 << pi)
                cand1 = _maximalize(cand1, all_indices, conflict)
                l1 = bin(cand1).count('1')
                # try adding just pj
                cand2 = base | (1 << pj)
                cand2 = _maximalize(cand2, all_indices, conflict)
                l2 = bin(cand2).count('1')
                # try adding both
                cand3 = base
                if (conflict[pi] & cand3) == 0:
                    cand3 |= (1 << pi)
                if (conflict[pj] & cand3) == 0:
                    cand3 |= (1 << pj)
                cand3 = _maximalize(cand3, all_indices, conflict)
                l3 = bin(cand3).count('1')
                # pick best
                if l1 <= l2 and l1 <= l3:
                    new_mask, nl = cand1, l1
                elif l2 <= l3:
                    new_mask, nl = cand2, l2
                else:
                    new_mask, nl = cand3, l3
                if nl < best_len:
                    current = new_mask
                    best = new_mask
                    best_len = nl
                    improved = True
                    break
                if nl < bin(current).count('1'):
                    current = new_mask
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 300  # hard wall-clock limit

    all_indices = list(range(NUM))
    all_points = list(itertools.product(range(N), repeat=2))

    # Precompute conflict table (one-time cost)
    if time.time() > deadline - 30:
        # extremely unlikely; fallback
        conflict, iso_pairs = _precompute()
    else:
        conflict, iso_pairs = _precompute()

    # Safety: if precompute took too long, still proceed with minimal search
    if time.time() > deadline - 20:
        cm = _greedy_conflict_order(all_indices, conflict)
        cm = _maximalize(cm, all_indices, conflict)
        return [all_points[i] for i in all_indices if (cm >> i) & 1]

    best_mask = None
    best_len = float('inf')

    def consider(mask):
        nonlocal best_mask, best_len
        mask = _maximalize(mask, all_indices, conflict)
        l = bin(mask).count('1')
        if l < best_len:
            best_len = l
            best_mask = mask

    # --- Phase 1: diverse greedy seeds ---
    rng = random.Random(123)
    orderings = [
        list(all_indices),
        sorted(all_indices, key=lambda p: (p // N, p % N)),
        sorted(all_indices, key=lambda p: (p % N, p // N)),
        sorted(all_indices, key=lambda p: (p // N + p % N, p // N)),
        sorted(all_indices, key=lambda p: bin(conflict[p]).count('1')),
    ]
    for order in orderings:
        if time.time() > deadline - 60:
            break
        consider(_greedy_order(order, conflict))

    consider(_greedy_conflict_order(all_indices, conflict))

    # randomized greedy restarts
    for _ in range(30):
        if time.time() > deadline - 60:
            break
        order = all_indices[:]
        rng.shuffle(order)
        consider(_greedy_order(order, conflict))

    # --- Phase 2: local search from best ---
    if best_mask is not None and time.time() < deadline - 30:
        ls = _local_search(best_mask, all_indices, conflict, iso_pairs,
                           max_iters=100000, deadline=deadline - 30, seed=1)
        consider(ls)

    # --- Phase 3: more local search from random restarts ---
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        order = all_indices[:]
        rng.shuffle(order)
        seed_mask = _greedy_order(order, conflict)
        ls = _local_search(seed_mask, all_indices, conflict, iso_pairs,
                           max_iters=30000, deadline=deadline - 15,
                           seed=100 + restart)
        consider(ls)

    # --- Phase 4: final polish ---
    if best_mask is not None and time.time() < deadline - 5:
        ls = _local_search(best_mask, all_indices, conflict, iso_pairs,
                           max_iters=50000, deadline=deadline - 5, seed=999)
        consider(ls)

    # Fallback
    if best_mask is None:
        best_mask = _maximalize(0, all_indices, conflict)

    result = [all_points[i] for i in all_indices if (best_mask >> i) & 1]

    # Final verification
    if not _is_valid_no_iso(result, iso_pairs):
        # rebuild from scratch
        best_mask = _maximalize(0, all_indices, conflict)
        result = [all_points[i] for i in all_indices if (best_mask >> i) & 1]

    if not _is_maximal(best_mask, all_indices, conflict):
        best_mask = _maximalize(best_mask, all_indices, conflict)
        result = [all_points[i] for i in all_indices if (best_mask >> i) & 1]

    return result
