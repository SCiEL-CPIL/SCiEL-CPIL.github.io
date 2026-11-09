"""Fresh design: randomized greedy + removal optimization for linear triple systems on 16 points."""
import random
import time

_START = time.time()
_LIMIT = 110.0  # safety margin well under the 120/150s kill

def _point_masks():
    return [1 << i for i in range(16)]

def _independence_number(triples, pm):
    # triples: list of (a,b,c) with a<b<c, all in 0..15
    # alpha = largest point set with no triple fully contained
    # A subset S is independent iff for every triple t, not (t[0] in S and t[1] in S and t[2] in S)
    # Equivalently: for every triple, the xor of its three point masks is NOT in the xor-set of S,
    # but that only checks "xor equals triple". We need: no triple's three points all in S.
    # Use the classic characterization: S independent iff for all t in triples,
    #   not (pm[t0] & pm[t1] & pm[t2] subset of mask(S)).
    # Efficient: iterate subsets by size descending.
    n = len(triples)
    # Precompute for each triple its three point masks
    tm = [(pm[a], pm[b], pm[c]) for (a, b, c) in triples]
    # Try sizes from 16 down
    from itertools import combinations
    for size in range(16, 0, -1):
        for combo in combinations(range(16), size):
            mask = 0
            for p in combo:
                mask |= pm[p]
            # check if any triple is fully contained
            ok = True
            for (ma, mb, mc) in tm:
                if (ma & mask) == ma and (mb & mask) == mb and (mc & mask) == mc:
                    ok = False
                    break
            if ok:
                return size
    return 0

def _is_valid(triples):
    # any two triples share at most one point
    for i in range(len(triples)):
        a, b, c = triples[i]
        for j in range(i + 1, len(triples)):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def _greedy_build(rng, pm, order_mode):
    # Build a linear triple system greedily.
    # order_mode: how to pick which triple to add next.
    points = list(range(16))
    rng.shuffle(points)
    triples = []
    # We'll maintain, for each point, the count of triples it's in, and a set of used point-pairs.
    used_pairs = set()
    # Build candidate pool incrementally.
    # Represent each point's incidence as a bitmask of which triples it's in? Simpler: track used pairs.
    # Greedy: repeatedly pick a triple that doesn't conflict, preferring triples covering "underused" points.
    # To keep it fast, precompute all C(16,3)=560 triples.
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
    # Shuffle candidate order
    cand = all_triples[:]
    rng.shuffle(cand)
    # We'll add triples until no more can be added (or a cap).
    # Track point usage count to bias.
    usage = [0] * 16
    max_rounds = 400
    added = 0
    while added < max_rounds:
        progressed = False
        # Re-shuffle remaining candidates each round is expensive; instead scan and pick a random valid one.
        # Scan candidates, collect valid ones, pick one.
        valid_now = []
        for t in cand:
            a, b, c = t
            # check pair conflicts
            if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
                continue
            valid_now.append(t)
        if not valid_now:
            break
        # pick one, biased toward covering underused points
        if order_mode == 0:
            t = valid_now[rng.randrange(len(valid_now))]
        else:
            # weight = sum of (16 - usage[p]) for p in t
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
        progressed = True
        if not progressed:
            break
    return triples

def _remove_to_optimize(triples, pm, rng, time_budget):
    # Given a valid system, try removing triples to reduce independence number.
    # alpha is monotone non-decreasing as we remove triples.
    # We want to remove triples that most reduce alpha.
    # Local search: try removing each triple, keep the one that gives best alpha, repeat.
    cur = list(triples)
    cur_alpha = _independence_number(cur, pm)
    improved = True
    start = time.time()
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_alpha = cur_alpha
        best_idx = -1
        n = len(cur)
        # Try removing each triple
        for i in range(n):
            # quick: if removing doesn't change alpha, skip heavy compute sometimes
            trial = cur[:i] + cur[i+1:]
            a = _independence_number(trial, pm)
            if a < best_alpha:
                best_alpha = a
                best_idx = i
                improved = True
            # early stop if we hit a very good value
        if improved:
            cur = cur[:best_idx] + cur[best_idx+1:]
            cur_alpha = best_alpha
    return cur, cur_alpha

def construct():
    rng = random.Random(12345)
    pm = _point_masks()
    best_alpha = 16
    best_triples = None
    start = time.time()

    # A few fixed heuristic order modes and random restarts
    order_modes = [0, 1]
    restarts = 0
    while (time.time() - start) < (_LIMIT - 20):
        mode = order_modes[restarts % len(order_modes)]
        triples = _greedy_build(rng, pm, mode)
        if not triples:
            restarts += 1
            continue
        # Post-optimization with a small time budget
        rem_budget = min(8.0, max(0.5, _LIMIT - 25 - (time.time() - start)))
        if rem_budget < 0.5:
            # Just evaluate as-is
            a = _independence_number(triples, pm)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
            restarts += 1
            continue
        opt_triples, a = _remove_to_optimize(triples, pm, rng, rem_budget)
        if a < best_alpha:
            best_alpha = a
            best_triples = opt_triples
        restarts += 1
        if best_alpha <= 6:
            break

    if best_triples is None:
        # Fallback: return the seed-like system (still valid)
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    # Final safety: ensure validity and within time
    if not _is_valid(best_triples):
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    return best_triples
