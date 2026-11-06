"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time

N = 16
DEADLINE = time.time() + 143  # safety margin before 150s kill

ALL_TRIPLES = []
TRIPLE_MASKS = {}
for t in itertools.combinations(range(N), 3):
    m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    ALL_TRIPLES.append(t)
    TRIPLE_MASKS[t] = m

PAIR_TO_TRIPLES = {}
for t in ALL_TRIPLES:
    a, b, c = t
    for p in ((a, b), (a, c), (b, c)):
        p = (min(p), max(p))
        PAIR_TO_TRIPLES.setdefault(p, []).append(t)

def triple_pairs(t):
    a, b, c = t
    return ((a, b) if a < b else (b, a),
            (a, c) if a < c else (c, a),
            (b, c) if b < c else (c, b))

def is_linear(triples):
    seen = set()
    for t in triples:
        for p in triple_pairs(t):
            if p in seen:
                return False
            seen.add(p)
    return True

def compute_alpha(triples):
    """Exact independence number via subset zeta transform. Fast for N=16."""
    if not triples:
        return N
    init = [False] * (1 << N)
    for t in triples:
        init[TRIPLE_MASKS[t]] = True
    has = init[:]
    for b in range(N):
        bit = 1 << b
        for S in range(1 << N):
            if S & bit and has[S ^ bit]:
                has[S] = True
    best = 0
    for S in range(1 << N):
        if not has[S]:
            pc = S.bit_count()
            if pc > best:
                best = pc
                if best == N:
                    break
    return best

def greedy_dense():
    order = ALL_TRIPLES[:]
    random.shuffle(order)
    triples = []
    used_pairs = set()
    for t in order:
        ps = triple_pairs(t)
        if not any(p in used_pairs for p in ps):
            triples.append(t)
            used_pairs.update(ps)
    return triples

def z2_construction():
    triples = []
    used_pairs = set()
    for x in range(16):
        for y in range(x + 1, 16):
            z = x ^ y
            if z == x or z == y:
                continue
            t = tuple(sorted((x, y, z)))
            ps = triple_pairs(t)
            if not any(p in used_pairs for p in ps):
                triples.append(t)
                used_pairs.update(ps)
    return triples

def cyclic_construction():
    base = (0, 1, 4)
    triples = []
    used_pairs = set()
    for shift in range(16):
        t = tuple(sorted(((base[0] + shift) % 16, (base[1] + shift) % 16, (base[2] + shift) % 16)))
        ps = triple_pairs(t)
        if not any(p in used_pairs for p in ps):
            triples.append(t)
            used_pairs.update(ps)
    return triples

def local_search(triples, max_iters=1500):
    if not is_linear(triples):
        return triples, compute_alpha(triples)
    cur_alpha = compute_alpha(triples)
    triples = list(triples)
    for _ in range(max_iters):
        if time.time() > DEADLINE - 6:
            break
        improved = False
        start = random.randrange(len(triples))
        for idx in range(len(triples)):
            if time.time() > DEADLINE - 6:
                break
            i = (start + idx) % len(triples)
            old = triples[i]
            old_pairs = triple_pairs(old)
            cands = []
            for p in old_pairs:
                cands.extend(PAIR_TO_TRIPLES[p])
            random.shuffle(cands)
            tried = 0
            for new in cands:
                if tried >= 6:
                    break
                tried += 1
                if new == old:
                    continue
                new_pairs = triple_pairs(new)
                conflict = False
                for j, ot in enumerate(triples):
                    if j == i:
                        continue
                    op = triple_pairs(ot)
                    for p in new_pairs:
                        if p in op:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    continue
                triples[i] = new
                na = compute_alpha(triples)
                if na < cur_alpha:
                    cur_alpha = na
                    improved = True
                    break
                else:
                    triples[i] = old
            if improved:
                break
        if not improved:
            break
    return triples, cur_alpha

def construct():
    best_triples = None
    best_alpha = N

    seeds = [z2_construction(), cyclic_construction(), greedy_dense(), greedy_dense(), greedy_dense()]

    for s in seeds:
        if not s or not is_linear(s):
            continue
        a = compute_alpha(s)
        if a < best_alpha:
            best_alpha = a
            best_triples = list(s)

    if best_triples is None:
        best_triples = greedy_dense()
        best_alpha = compute_alpha(best_triples)

    if time.time() < DEADLINE - 8:
        best_triples, best_alpha = local_search(best_triples, max_iters=2000)

    while time.time() < DEADLINE - 12:
        t = greedy_dense()
        if not t:
            break
        t, a = local_search(t, max_iters=600)
        if a < best_alpha:
            best_alpha = a
            best_triples = t

    if best_triples is None:
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    return best_triples
