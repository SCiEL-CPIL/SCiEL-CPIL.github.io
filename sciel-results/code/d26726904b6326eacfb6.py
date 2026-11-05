"""Two-colourings without a monochromatic K_{2,2,2}; simulated annealing + full validation."""
import time
import random

_DEADLINE_PAD = 108  # return well before the 120s / 150s limits

# ---- precompute the 15 partitions of 6 positions into 3 pairs, and their 12 cross pairs ----
def _build_pos_parts():
    matchings = []
    for a in range(1, 6):
        R = [x for x in range(1, 6) if x != a]
        R.sort()
        s = R[0]
        for c in R[1:]:
            R2 = [x for x in R if x != s and x != c]
            d1, d2 = R2
            matchings.append(((0, a), (s, c), (d1, d2)))
    pos_parts = []
    for m in matchings:
        cross = []
        pairs = list(m)
        for x in range(3):
            for y in range(x + 1, 3):
                for u in pairs[x]:
                    for v in pairs[y]:
                        cross.append((u, v))
        pos_parts.append(cross)  # 12 (u,v) position pairs
    return pos_parts

_POS_PARTS = _build_pos_parts()  # 15 x 12


def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    cols = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': cols}


def _full_valid(n, colors):
    """colors: list length n*n, symmetric. True iff no mono K_{2,2,2}."""
    import itertools
    for six in itertools.combinations(range(n), 6):
        V = six
        for cross in _POS_PARTS:
            # check all-0
            c0 = colors[V[cross[0][0]] * n + V[cross[0][1]] if V[cross[0][0]] < V[cross[0][1]]
                     else V[cross[0][1]] * n + V[cross[0][0]]]
            all0 = True
            all1 = True
            for (u, v) in cross:
                a, b = V[u], V[v]
                lin = a * n + b if a < b else b * n + a
                cv = colors[lin]
                if cv != 0:
                    all0 = False
                if cv != 1:
                    all1 = False
                if not all0 and not all1:
                    break
            if all0 or all1:
                return False
    return True


def _search(n, time_budget, rng):
    """Return a valid color list for K_n or None."""
    nn = n * n
    colors = [rng.randint(0, 1) for _ in range(nn)]
    # symmetrize
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[i * n + j]
            colors[j * n + i] = c

    S = 3000
    sets = []
    part_edges = []  # [s][p] = list of 12 canonical edge indices
    edge_inc = {}
    for s in range(S):
        V = rng.sample(range(n), 6)
        sets.append(V)
        pe = []
        for cross in _POS_PARTS:
            lst = []
            for (u, v) in cross:
                a, b = V[u], V[v]
                lst.append(a * n + b if a < b else b * n + a)
            pe.append(lst)
        part_edges.append(pe)
        for p in range(15):
            for lin in pe[p]:
                edge_inc.setdefault(lin, []).append((s, p))

    def is_mono(s, p):
        arr = part_edges[s][p]
        c = colors[arr[0]]
        for k in range(1, 12):
            if colors[arr[k]] != c:
                return False
        return True

    # initial objective
    obj = 0
    for s in range(S):
        for p in range(15):
            if is_mono(s, p):
                obj += 1

    def flip_delta(i, j):
        lin = i * n + j
        sym = j * n + i
        inc = edge_inc.get(lin)
        if not inc:
            return 0
        d = 0
        for (s, p) in inc:
            before = is_mono(s, p)
            colors[lin] = 1 - colors[lin]
            colors[sym] = colors[lin]
            after = is_mono(s, p)
            colors[lin] = 1 - colors[lin]
            colors[sym] = colors[lin]
            d += (1 if after else 0) - (1 if before else 0)
        return d

    def do_flip(i, j):
        nonlocal obj
        lin = i * n + j
        sym = j * n + i
        inc = edge_inc.get(lin)
        if not inc:
            return
        d = 0
        for (s, p) in inc:
            before = is_mono(s, p)
            colors[lin] = 1 - colors[lin]
            colors[sym] = colors[lin]
            after = is_mono(s, p)
            d += (1 if after else 0) - (1 if before else 0)
        obj += d

    t0 = time.time()
    T = 3.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    step = 0
    while time.time() - t0 < time_budget:
        i, j = rng.choice(pairs)
        d = flip_delta(i, j)
        if d <= 0 or rng.random() < pow(2.718281828, -d / T):
            do_flip(i, j)
        step += 1
        if step % 200 == 0:
            T = max(0.05, T * 0.999)
            if obj == 0:
                break
        if step % 5000 == 0 and obj > 0:
            # focused repair: find a bad (s,p), flip a random edge in it if it helps
            bads = [(s, p) for s in range(S) for p in range(15) if is_mono(s, p)]
            if bads:
                s, p = rng.choice(bads)
                arr = part_edges[s][p]
                lin = rng.choice(arr)
                a = lin // n
                b = lin % n
                if a > b:
                    a, b = b, a
                d2 = flip_delta(a, b)
                if d2 <= 0:
                    do_flip(a, b)

    if obj == 0:
        # sample clean -> full validate
        if _full_valid(n, colors):
            return [colors[i * n + j] for i in range(n) for j in range(i + 1, n)]
    return None


def construct():
    start = time.time()
    deadline = start + _DEADLINE_PAD
    rng = random.Random(12345)
    best = _paley17()
    best_n = 17
    for n in range(29, 17, -1):
        if time.time() > deadline - 4:
            break
        remaining = deadline - time.time()
        alloc = min(remaining - 2, 22.0)
        if alloc < 1.5:
            break
        res = _search(n, alloc, rng)
        if res is not None:
            return {'n': n, 'colors': res}
        if n > best_n:
            best_n = n
    if time.time() > deadline - 2:
        return best
    return best
