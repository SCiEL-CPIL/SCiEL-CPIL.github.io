"""Combined program: fast bitmask independence + greedy build + removal refinement."""
import time
import random
import math
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # safety margin well under the 150s kill

def _point_masks():
    return [1 << i for i in range(16)]

def _independence_number(triples, pm):
    """Exact independence number via bitmask + descending-size search.

    A subset S is independent iff no triple is fully contained in S.
    We try sizes from 16 down and return the first size with a valid S.
    """
    n = len(triples)
    if n == 0:
        return 16
    # Precompute each triple as a 16-bit mask
    tmask = []
    for (a, b, c) in triples:
        m = pm[a] | pm[b] | pm[c]
        tmask.append(m)
    # Try sizes from 16 down to 1
    for size in range(16, 0, -1):
        for combo in combinations(range(16), size):
            mask = 0
            for p in combo:
                mask |= pm[p]
            # check if any triple is fully contained
            ok = True
            for m in tmask:
                if (m & mask) == m:
                    ok = False
                    break
            if ok:
                return size
    return 0

def _is_valid(triples):
    """Check if any two triples share at most one point (linear)."""
    n = len(triples)
    for i in range(n):
        a, b, c = triples[i]
        for j in range(i + 1, n):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z:
                inter += 1
            if b == x or b == y or b == z:
                inter += 1
            if c == x or c == y or c == z:
                inter += 1
            if inter > 1:
                return False
    return True

def _greedy_build(rng, pm, order_mode):
    """Build a linear triple system greedily with pair-conflict tracking."""
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
    cand = all_triples[:]
    rng.shuffle(cand)
    triples = []
    used_pairs = set()
    usage = [0] * 16
    max_rounds = 400
    added = 0
    while added < max_rounds:
        valid_now = []
        for t in cand:
            a, b, c = t
            if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
                continue
            valid_now.append(t)
        if not valid_now:
            break
        if order_mode == 0:
            t = valid_now[rng.randrange(len(valid_now))]
        else:
            best = None
            best_w = -1
            for t in valid_now:
                w = (16 - usage[t[0]]) + (16 - usage[t[1]]) + (16 - usage[t[2]])
                if w > best_w:
                    best_w = w
                    best = t
            t = best
        a, b, c = t
        triples.append(t)
        used_pairs.add((a, b)); used_pairs.add((a, c)); used_pairs.add((b, c))
        usage[a] += 1; usage[b] += 1; usage[c] += 1
        added += 1
    return triples

def _remove_to_optimize(triples, pm, time_budget):
    """Remove triples to reduce the independence number."""
    cur = list(triples)
    cur_alpha = _independence_number(cur, pm)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_alpha = cur_alpha
        best_idx = -1
        n = len(cur)
        for i in range(n):
            trial = cur[:i] + cur[i + 1:]
            a = _independence_number(trial, pm)
            if a < best_alpha:
                best_alpha = a
                best_idx = i
                improved = True
        if improved:
            cur = cur[:best_idx] + cur[best_idx + 1:]
            cur_alpha = best_alpha
    return cur, cur_alpha

def construct():
    rng = random.Random(12345)
    pm = _point_masks()
    best_alpha = 16
    best_triples = None
    start = time.time()

    order_modes = [0, 1]
    restarts = 0
    while (time.time() - start) < (_LIMIT - 20):
        mode = order_modes[restarts % len(order_modes)]
        triples = _greedy_build(rng, pm, mode)
        if not triples:
            restarts += 1
            continue
        rem_budget = min(8.0, max(0.5, _LIMIT - 25 - (time.time() - start)))
        if rem_budget < 0.5:
            a = _independence_number(triples, pm)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
            restarts += 1
            continue
        opt_triples, a = _remove_to_optimize(triples, pm, rem_budget)
        if a < best_alpha:
            best_alpha = a
            best_triples = opt_triples
        restarts += 1
        if best_alpha <= 6:
            break

    if best_triples is None:
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    if not _is_valid(best_triples):
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return [list(t) for t in best_triples]
