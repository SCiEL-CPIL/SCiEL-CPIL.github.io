import random
import time
import math
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
N = 16
POINTS = list(range(N))
FULL_MASK = (1 << N) - 1
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLE_MASKS = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in ALL_TRIPLES]


def _exact_alpha(triples):
    """Exact independence number via mask enumeration. Returns (alpha, witness_mask)."""
    if not triples:
        return N, FULL_MASK
    t_masks = list(set(TRIPLE_MASKS[ALL_TRIPLES.index(t)] for t in triples))
    best_a = 0
    best_m = 0
    for mask in range(FULL_MASK + 1):
        pc = mask.bit_count()
        if pc <= best_a:
            continue
        indep = True
        for tm in t_masks:
            if (tm & mask) == tm:
                indep = False
                break
        if indep:
            best_a = pc
            best_m = mask
    return best_a, best_m


def _greedy_alpha(triples):
    """Fast heuristic (lower bound) independence number."""
    if not triples:
        return N
    t_set = set(frozenset(t) for t in triples)
    deg = defaultdict(int)
    for t in triples:
        for p in t:
            deg[p] += 1
    chosen = set()
    for p in sorted(POINTS, key=lambda x: deg[x]):
        ok = True
        for q in chosen:
            for r in chosen:
                if q != r and frozenset((p, q, r)) in t_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.add(p)
    return len(chosen)


def _sa_search(deadline, seed):
    rng = random.Random(seed)

    def init_system():
        used_pairs = set()
        triples = []
        cand = ALL_TRIPLES[:]
        rng.shuffle(cand)
        for t in cand:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            if all(p not in used_pairs for p in pairs):
                triples.append(t)
                used_pairs.update(pairs)
        return triples, used_pairs

    triples, used_pairs = init_system()
    cur_alpha = _greedy_alpha(triples)
    best_alpha = cur_alpha
    best_triples = list(triples)

    t = 2.0
    t_min = 0.02
    cool = 0.9995
    step = 0
    while time.time() < deadline:
        step += 1
        # snapshot for revert
        snap_triples = list(triples)
        snap_pairs = set(used_pairs)

        if len(triples) > 0:
            i = rng.randrange(len(triples))
            rem_t = triples[i]
            a, b, c = rem_t
            for p in [(a, b), (a, c), (b, c)]:
                used_pairs.discard(p)
            triples.pop(i)
            # try to add a random valid triple
            cand = ALL_TRIPLES[:]
            rng.shuffle(cand)
            for tt in cand:
                x, y, z = tt
                pairs = [(x, y), (x, z), (y, z)]
                if all(p not in used_pairs for p in pairs):
                    triples.append(tt)
                    used_pairs.update(pairs)
                    break
        else:
            tt = rng.choice(ALL_TRIPLES)
            x, y, z = tt
            used_pairs.update([(x, y), (x, z), (y, z)])
            triples.append(tt)

        new_alpha = _greedy_alpha(triples)
        if new_alpha < cur_alpha:
            cur_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = list(triples)
        else:
            delta = new_alpha - cur_alpha
            if rng.random() < math.exp(-delta / max(t, 1e-6)):
                cur_alpha = new_alpha
            else:
                triples = snap_triples
                used_pairs = snap_pairs

        t = max(t_min, t * cool)

        if step % 300 == 0:
            ea, _ = _exact_alpha(best_triples)
            if ea < best_alpha:
                best_alpha = ea

    return best_alpha, best_triples


def _polish(triples, deadline, seed):
    rng = random.Random(seed)
    cur = list(triples)
    cur_alpha, _ = _exact_alpha(cur)
    best = list(cur)
    best_alpha = cur_alpha

    def used_pairs_of(ts):
        up = set()
        for t in ts:
            a, b, c = t
            up.update([(a, b), (a, c), (b, c)])
        return up

    def fill(ts, up):
        changed = True
        while changed:
            changed = False
            cand = ALL_TRIPLES[:]
            rng.shuffle(cand)
            for t in cand:
                x, y, z = t
                pairs = [(x, y), (x, z), (y, z)]
                if all(p not in up for p in pairs):
                    ts.append(t)
                    up.update(pairs)
                    changed = True
                    break
        return ts, up

    cur, up = fill(cur, used_pairs_of(cur))
    cur_alpha, _ = _exact_alpha(cur)
    if cur_alpha < best_alpha:
        best_alpha = cur_alpha
        best = list(cur)

    for _ in range(3000):
        if time.time() > deadline:
            break
        if len(cur) == 0:
            break
        i = rng.randrange(len(cur))
        rem_t = cur[i]
        a, b, c = rem_t
        for p in [(a, b), (a, c), (b, c)]:
            up.discard(p)
        cur.pop(i)
        cur, up = fill(cur, up)
        cur_alpha, _ = _exact_alpha(cur)
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best = list(cur)
            cur = list(best)
            up = used_pairs_of(cur)
            cur, up = fill(cur, up)
        else:
            cur = list(best)
            up = used_pairs_of(cur)
            break

    return best_alpha, best


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_alpha = N
    best_triples = []

    seeds = [12345, 67890, 24680, 13579, 97531, 11111, 22222, 33333]
    for s in seeds:
        if time.time() > deadline - 40:
            break
        a, trs = _sa_search(deadline - 40, s)
        if a < best_alpha:
            best_alpha = a
            best_triples = trs
        if best_alpha <= 7:
            break

    if time.time() < deadline - 15 and best_triples:
        a, trs = _polish(best_triples, deadline - 10, 42)
        if a < best_alpha:
            best_alpha = a
            best_triples = trs

    if best_triples:
        ea, _ = _exact_alpha(best_triples)
        best_alpha = ea

    used = set()
    valid = []
    for t in best_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        if all(p not in used for p in pairs):
            valid.append(t)
            used.update(pairs)
    return valid
# EVOLVE-BLOCK-END
