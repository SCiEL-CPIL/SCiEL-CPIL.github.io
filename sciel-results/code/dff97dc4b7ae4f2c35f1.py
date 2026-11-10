import time
import random
from itertools import combinations

DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill

def _now():
    return time.time()

def _is_linear(triples):
    # check pairwise intersection size <= 1
    for i in range(len(triples)):
        a = set(triples[i])
        for j in range(i + 1, len(triples)):
            b = set(triples[j])
            if len(a & b) > 1:
                return False
    return True

def _independence_number(triples, points=16):
    # exact via branch and bound over subsets (16 points => 65536 subsets, feasible)
    n = points
    # precompute for each subset whether it contains a triple
    # represent triple as bitmask
    masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        masks.append(m)
    # a subset S is independent iff no triple is fully contained in S
    # We want max |S| with no triple contained.
    # DP / branch: iterate all subsets? 2^16 = 65536, for each check containment.
    # Precompute for each triple mask, mark all supersets as "not independent".
    bad = [False] * (1 << n)
    for m in masks:
        # mark all supersets of m
        # supersets: for each free bit, set it
        free = ((1 << n) - 1) ^ m
        sub = free
        # iterate all subsets of free, sup = m | sub
        # number of supersets = 2^(popcount(free))
        s = sub
        while True:
            bad[m | s] = True
            if s == 0:
                break
            s = (s - 1) & sub
    best = 0
    for S in range(1 << n):
        if not bad[S]:
            pc = S.bit_count()
            if pc > best:
                best = pc
    return best

def _greedy_maximal_independent_set(triples, points=16, rng=None):
    # return a large independent set (not necessarily maximum)
    n = points
    # candidate points
    avail = list(range(n))
    # randomize order
    if rng is not None:
        rng.shuffle(avail)
    chosen = []
    chosen_set = set()
    for p in avail:
        # check that no triple is fully contained in chosen + {p}
        ok = True
        for t in triples:
            # triple t is contained in chosen_set | {p} iff all of t in that set
            if all(x in chosen_set or x == p for x in t):
                ok = False
                break
        if ok:
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _random_greedy_packing(rng, points=16, target_triples=None):
    # build a linear triple system by greedily adding random triples
    n = points
    triples = []
    # precompute all candidate triples
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)
    # track pairwise intersection counts to quickly reject
    # for speed, maintain a set of pairs that are "used" by at most one triple? 
    # Actually a pair can appear in multiple triples as long as each pair of triples shares <=1 point.
    # Two triples share >1 point iff they share a pair. So we need: no pair appears in two triples.
    # That's the linearity condition for triples: any two triples intersect in at most one point <=> no pair is in two triples.
    used_pairs = set()
    for t in all_triples:
        pairs = list(combinations(t, 2))
        if any(p in used_pairs for p in pairs):
            continue
        # accept
        for p in pairs:
            used_pairs.add(p)
        triples.append(list(t))
        if target_triples is not None and len(triples) >= target_triples:
            break
    return triples

def _try_add_triple_breaking_iset(triples, iset, rng, points=16):
    # find a triple that hits iset and is linear with all existing triples
    n = points
    iset_set = set(iset)
    # candidate triples that contain at least one point of iset
    # to keep it fast, sample
    candidates = []
    for a in iset:
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                candidates.append((a, b, c))
    rng.shuffle(candidates)
    # dedupe
    seen = set()
    for t in candidates:
        if t in seen:
            continue
        seen.add(t)
        # check linear with existing triples
        ok = True
        for ex in triples:
            inter = len(set(t) & set(ex))
            if inter > 1:
                ok = False
                break
        if ok:
            return list(t)
    return None

def _remove_triple_redundant(triples, rng, points=16):
    # find a triple whose removal still allows the system to remain valid (always valid)
    # and ideally we can then add more. For now, just randomly remove one with low "coverage"
    if not triples:
        return None, None
    # choose a triple that shares few pairs with others (i.e., removing it frees pairs)
    # score = number of pairs it uses (always 3) but also how many other triples would become addable
    # simple: random
    idx = rng.randrange(len(triples))
    t = triples[idx]
    del triples[idx]
    return t, idx

def construct():
    start = _now()
    deadline = start + 150.0 - DEADLINE_MARGIN
    rng = random.Random(12345)

    n = 16
    best_triples = None
    best_alpha = n + 1

    def consider(triples):
        nonlocal best_triples, best_alpha
        if not _is_linear(triples):
            return
        a = _independence_number(triples, n)
        if a < best_alpha:
            best_alpha = a
            best_triples = [list(t) for t in triples]

    # Phase 1: many random greedy packings
    attempts = 0
    while _now() < deadline:
        attempts += 1
        triples = _random_greedy_packing(rng, n)
        consider(triples)
        # local search: try to break independent sets
        for _ in range(200):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                # can't improve via this set
                continue
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                # try removing a random triple then adding
                removed, _ = _remove_triple_redundant(triples, rng, n)
                if removed is not None:
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                    else:
                        triples.append(removed)
                break
        if attempts >= 200:
            break

    # Phase 2: targeted improvement on best
    if best_triples is not None and _now() < deadline:
        triples = [list(t) for t in best_triples]
        for _ in range(500):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                break
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                # try a few random removals then add
                improved = False
                for _ in range(10):
                    removed, _ = _remove_triple_redundant(triples, rng, n)
                    if removed is None:
                        break
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                        improved = True
                        break
                    else:
                        triples.append(removed)
                if not improved:
                    break

    # ensure we always return a valid system
    if best_triples is None:
        # fallback: seed
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best_triples
