"""Fresh design: randomized greedy + removal optimization for linear triple systems on 16 points."""
import random
import time

_START = time.time()
_LIMIT = 110.0  # safety margin well under the 120/150s kill

def _point_masks():
    return [1 << i for i in range(16)]

def _independence_number(triples, pm):
    """Compute independence number using bitmask subset enumeration with early termination."""
    if not triples:
        return 16
    # Precompute triple masks: each triple as a single 16-bit mask
    triple_masks = []
    for (a, b, c) in triples:
        tm = pm[a] | pm[b] | pm[c]
        triple_masks.append(tm)
    
    # For each subset S (bitmask), check if it's independent:
    # S is independent iff for all t in triple_masks, (S & t) != t
    # We want the largest |S| such that S is independent.
    # Enumerate by size descending.
    from itertools import combinations
    for size in range(16, 0, -1):
        for combo in combinations(range(16), size):
            mask = 0
            for p in combo:
                mask |= pm[p]
            # Check no triple is fully contained in mask
            independent = True
            for tmask in triple_masks:
                if (mask & tmask) == tmask:
                    independent = False
                    break
            if independent:
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
    """Local search: try removing triples to reduce independence number.
    
    Key optimization: alpha only increases when we remove triples, so we're looking
    for the MINIMUM alpha achievable. We use a greedy approach: try removing each triple,
    keep the one that gives the best (lowest) alpha.
    
    Since alpha can only increase (or stay same) when removing triples, and we want to
    minimize alpha, we actually want to KEEP triples that help reduce alpha.
    
    Wait - re-reading: alpha is the LARGEST independent set. More triples = smaller alpha.
    So removing triples INCREASES alpha. We want to minimize alpha, so we should NOT remove
    triples unless it helps... but it can't help since removing increases alpha.
    
    Actually the original code seems confused. Let me re-read the task:
    "score is the largest point set containing no triple" = alpha
    "Minimize" alpha.
    
    So more triples = smaller alpha (good). We want to ADD triples, not remove them.
    The greedy build already adds triples until no more can be added.
    
    So _remove_to_optimize is actually counterproductive! Let me replace it with
    a swap-based local search: try replacing one triple with another to reduce alpha.
    """
    cur = list(triples)
    if not cur:
        return cur, 16
    cur_alpha = _independence_number(cur, pm)
    start = time.time()
    improved = True
    # Precompute all valid triples
    all_triples = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                all_triples.append((i, j, k))
    
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_alpha = cur_alpha
        best_move = None  # (remove_idx, add_triple)
        n = len(cur)
        # Build used pairs
        used_pairs = set()
        for (a, b, c) in cur:
            used_pairs.add((a, b)); used_pairs.add((a, c)); used_pairs.add((b, c))
        
        # Try swapping: remove one triple, add another
        for i in range(n):
            a, b, c = cur[i]
            # Temporarily remove
            trial_used = set()
            for j in range(n):
                if j == i:
                    continue
                x, y, z = cur[j]
                trial_used.add((x, y)); trial_used.add((x, z)); trial_used.add((y, z))
            
            # Find candidate triples to add
            # Sample a few candidates for speed
            candidates = []
            for t in all_triples:
                ta, tb, tc = t
                if (ta, tb) in trial_used or (ta, tc) in trial_used or (tb, tc) in trial_used:
                    continue
                candidates.append(t)
            
            if not candidates:
                continue
            
            # Try a random sample of candidates
            sample_size = min(20, len(candidates))
            sampled = rng.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
            
            for new_t in sampled:
                trial = cur[:i] + cur[i+1:] + [new_t]
                a_val = _independence_number(trial, pm)
                if a_val < best_alpha:
                    best_alpha = a_val
                    best_move = (i, new_t)
                    improved = True
        
        if best_move is not None:
            remove_idx, add_t = best_move
            cur = cur[:remove_idx] + cur[remove_idx+1:] + [add_t]
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
        opt_budget = min(10.0, max(0.5, _LIMIT - 25 - (time.time() - start)))
        if opt_budget < 0.5:
            # Just evaluate as-is
            a = _independence_number(triples, pm)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
            restarts += 1
            continue
        opt_triples, a = _remove_to_optimize(triples, pm, rng, opt_budget)
        if a < best_alpha:
            best_alpha = a
            best_triples = opt_triples
        restarts += 1
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback: return the seed-like system (still valid)
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    # Final safety: ensure validity and within time
    if not _is_valid(best_triples):
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    return best_triples
