"""Rewritten solver: randomized greedy + 1-out/2-in local search with deadline guard."""
import time
import random
import itertools

_N = 128

def _all_triples():
    # Precompute all (a,b,c) with 1<=a<b<c<=128
    tr = []
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            for c in range(b + 1, _N + 1):
                tr.append((a, b, c))
    return tr

def _triple_sums(triple):
    a, b, c = triple
    return (a + b, a + c, b + c)

def _build_membership(S):
    return [False] * (_N + 1) | {x: True for x in S} if False else (lambda: None)()

def _membership(S):
    m = [False] * (_N + 1)
    for x in S:
        if 1 <= x <= _N:
            m[x] = True
    return m

def _triple_violates(triple, m):
    s = _triple_sums(triple)
    return m[s[0]] and m[s[1]] and m[s[2]]

def _is_valid(S):
    m = _membership(S)
    for t in _ALL_TRIPLES:
        if _triple_violates(t, m):
            return False
    return True

def _violating_triples(S):
    m = _membership(S)
    out = []
    for t in _ALL_TRIPLES:
        if _triple_violates(t, m):
            out.append(t)
    return out

def _greedy_add(S, order, deadline):
    # S is a set; try adding each element in order if it keeps validity
    S = set(S)
    m = _membership(S)
    # maintain count of currently-violating triples for speed? We'll re-check incrementally.
    # For each candidate x, check whether adding x creates a violation.
    # A violation involving x: x is one of a,b,c OR x is one of the sums.
    # Precompute for each x the list of triples where x is a member, and where x is a sum.
    for x in order:
        if time.time() > deadline:
            break
        if x in S:
            continue
        # candidate: add x
        # check triples where x is a member
        ok = True
        for t in _MEM_TRIPLES[x]:
            a, b, c = t
            if not (a in S and b in S and c in S):
                continue
            s = _triple_sums(t)
            # need all three sums in S (x is not a sum here since x is a member)
            if m[s[0]] and m[s[1]] and m[s[2]]:
                ok = False
                break
        if ok:
            # check triples where x is one of the sums
            for t in _SUM_TRIPLES[x]:
                a, b, c = t
                if not (a in S and b in S and c in S):
                    continue
                s = _triple_sums(t)
                if m[s[0]] and m[s[1]] and m[s[2]]:
                    ok = False
                    break
        if ok:
            S.add(x)
            m[x] = True
    return S

def _local_search(S, deadline, rng):
    # 1-out/2-in: remove one element r, add two elements a,b not in S (a!=b, a!=r, b!=r)
    # if the result is valid and larger (net +1), accept.
    S = set(S)
    best = S
    best_size = len(S)
    elements = list(range(1, _N + 1))
    # Precompute pairs of additions
    # We'll iterate removals and candidate pairs, but that's expensive; cap attempts.
    attempts = 0
    max_attempts = 200000
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        r = rng.choice(elements)
        if r not in S:
            continue
        S2 = S - {r}
        # pick two distinct elements not in S2
        avail = [x for x in elements if x not in S2]
        if len(avail) < 2:
            continue
        a, b = rng.sample(avail, 2)
        cand = S2 | {a, b}
        if _is_valid(cand):
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
                S = cand  # continue from improved state
    return best

_ALL_TRIPLES = _all_triples()
# Index triples by member and by sum
_MEM_TRIPLES = {x: [] for x in range(1, _N + 1)}
_SUM_TRIPLES = {x: [] for x in range(2, 2 * _N + 1)}
for t in _ALL_TRIPLES:
    a, b, c = t
    _MEM_TRIPLES[a].append(t)
    _MEM_TRIPLES[b].append(t)
    _MEM_TRIPLES[c].append(t)
    for s in _triple_sums(t):
        if s <= 2 * _N:
            _SUM_TRIPLES[s].append(t)

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin under 60s
    rng = random.Random(12345)

    best = set(range(_N // 2 + 1, _N + 1))  # seed: 65..128, size 64
    best_size = len(best)

    # Phase 1: multiple randomized greedy runs
    # Orders: random, descending, ascending, and a few mixes
    orders = []
    base = list(range(1, _N + 1))
    orders.append(list(reversed(base)))            # descending
    orders.append(base)                            # ascending
    for _ in range(6):
        o = base[:]
        rng.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        S = set(range(_N // 2 + 1, _N + 1))
        S = _greedy_add(S, order, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Phase 2: local search from best
    best = _local_search(best, deadline, rng)
    best_size = len(best)

    # Phase 3: a few more greedy+local iterations from different seeds
    for _ in range(4):
        if time.time() > deadline:
            break
        o = list(range(1, _N + 1))
        rng.shuffle(o)
        # start from a random subset of size ~40 to diversify
        start = set(rng.sample(range(1, _N + 1), 40))
        S = _greedy_add(start, o, deadline)
        S = _local_search(S, deadline, rng)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    result = sorted(best)
    # Validate once
    if not _is_valid(result):
        # fallback to seed
        result = list(range(_N // 2 + 1, _N + 1))
    return [int(x) for x in result]
