import time, random

_START = time.time()
_DEADLINE = _START + 70.0


def _max_sidon_size(S):
    n = len(S)
    # greedy lower bound
    greedy = []
    gsums = set()
    for v in S:
        ok = True
        for c in greedy:
            if (v + c) in gsums:
                ok = False
                break
        if ok and (v + v) in gsums:
            ok = False
        if ok:
            for c in greedy:
                gsums.add(v + c)
            gsums.add(v + v)
            greedy.append(v)
    best = [len(greedy)]

    def dfs(pos, chosen, sums, size):
        if size + (n - pos) <= best[0]:
            return
        if pos == n:
            if size > best[0]:
                best[0] = size
            return
        dfs(pos + 1, chosen, sums, size)
        v = S[pos]
        ok = True
        for c in chosen:
            if (v + c) in sums:
                ok = False
                break
        if ok and (v + v) in sums:
            ok = False
        if ok:
            ns = set(sums)
            for c in chosen:
                ns.add(v + c)
            ns.add(v + v)
            dfs(pos + 1, chosen + [v], ns, size + 1)

    dfs(0, [], set(), 0)
    return best[0]


def _make_pair_structured(rng):
    S = []
    used = set()
    sums = rng.sample(range(4, 256), 6)
    for s in sums:
        cands = [a for a in range(0, s // 2 + 1) if a != s - a and 0 <= s - a <= 128]
        rng.shuffle(cands)
        placed = False
        for a in cands:
            b = s - a
            if a not in used and b not in used:
                S.append(a); S.append(b); used.add(a); used.add(b); placed = True
                break
        if not placed:
            for v in rng.sample(range(129), 129):
                if v not in used:
                    S.append(v); used.add(v)
                    break
    while len(S) < 12:
        v = rng.randint(0, 128)
        if v not in used:
            S.append(v); used.add(v)
    return S[:12]


def _local_search(S, rng, budget):
    cur = list(S)
    cur_val = _max_sidon_size(cur)
    best = list(cur)
    best_val = cur_val
    steps = 0
    while time.time() < budget and steps < 300:
        steps += 1
        i = rng.randrange(12)
        old = cur[i]
        new = rng.randint(0, 128)
        if new == old or new in set(cur) - {old}:
            continue
        cur[i] = new
        val = _max_sidon_size(cur)
        if val < cur_val:
            cur_val = val
            if val < best_val:
                best_val = val
                best = list(cur)
        else:
            cur[i] = old
    return best, best_val


def construct():
    rng = random.Random(20240517)
    global_best = None
    global_val = 13
    S0 = _make_pair_structured(rng)
    S0, v0 = _local_search(S0, rng, _DEADLINE)
    global_best, global_val = S0, v0
    while time.time() < _DEADLINE:
        S = _make_pair_structured(rng)
        S, v = _local_search(S, rng, _DEADLINE)
        if v < global_val:
            global_val = v
            global_best = S
    if global_best is None or len(set(global_best)) != 12 or any(x < 0 or x > 128 for x in global_best):
        global_best = list(range(12))
    return global_best
