import math
import time
import random
from collections import deque

MAX_MOD = 128
DEADLINE = 90.0
MARGIN = 3.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _build(mods):
    n = len(mods)
    idx = {m: i for i, m in enumerate(mods)}
    # gcd matrix
    gcd_m = [[0] * n for _ in range(n)]
    # residue-class sets per (i,j): set of residue classes mod g that class i's residue must NOT equal
    forbidden = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                g = math.gcd(mods[i], mods[j])
                gcd_m[i][j] = g
                forbidden[i][j] = set()
    return n, idx, gcd_m, forbidden

def _ac3(n, mods, domains, gcd_m, forbidden, time_limit, step_budget):
    # arc (i,j) means: for every r in domains[i], there must exist s in domains[j]
    # with (s - r) % gcd != 0.
    queue = deque()
    for i in range(n):
        for j in range(n):
            if i != j:
                queue.append((i, j))
    # track which arcs are still "dirty" to avoid duplicates
    inq = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                inq[i][j] = True
    steps = 0
    while queue:
        if time.time() > time_limit:
            return None
        if steps > step_budget:
            return None
        steps += 1
        i, j = queue.popleft()
        inq[i][j] = False
        g = gcd_m[i][j]
        dj = domains[j]
        # for each r in domains[i], check support in dj
        new_di = []
        removed = False
        for r in domains[i]:
            # need some s in dj with (s - r) % g != 0
            ok = False
            for s in dj:
                if (s - r) % g != 0:
                    ok = True
                    break
            if ok:
                new_di.append(r)
            else:
                removed = True
        if removed:
            domains[i] = new_di
            if len(new_di) == 0:
                return None
            # re-add arcs (k,i) for k != i, k != j
            for k in range(n):
                if k != i and k != j and not inq[k][i]:
                    queue.append((k, i))
                    inq[k][i] = True
    return domains

def _solve_csp(mods, time_limit, step_budget):
    n, idx, gcd_m, forbidden = _build(mods)
    domains = [set(range(mods[i])) for i in range(n)]
    res = _ac3(n, mods, domains, gcd_m, forbidden, time_limit, step_budget)
    if res is None:
        return None
    domains = res
    assignment = [None] * n

    # precompute for speed: for each i, list of (j, g) neighbors
    # MRV: pick unassigned var with smallest domain
    def backtrack():
        if time.time() > time_limit:
            return 'timeout'
        # find unassigned with min domain
        best_i = -1
        best_size = 10**9
        for i in range(n):
            if assignment[i] is None:
                sz = len(domains[i])
                if sz == 0:
                    return False
                if sz < best_size:
                    best_size = sz
                    best_i = i
        if best_i == -1:
            return True
        # values: prefer those that keep most support (least constraining)
        vals = list(domains[best_i])
        random.shuffle(vals)
        # forward check
        for r in vals:
            assignment[best_i] = r
            # check all unassigned neighbors have support
            feasible = True
            saved = []
            for j in range(n):
                if j == best_i or assignment[j] is not None:
                    continue
                g = gcd_m[best_i][j]
                new_dj = []
                for s in domains[j]:
                    if (s - r) % g != 0:
                        new_dj.append(s)
                if len(new_dj) == 0:
                    feasible = False
                    break
                saved.append((j, domains[j], new_dj))
            if not feasible:
                assignment[best_i] = None
                continue
            for (j, old, new) in saved:
                domains[j] = new
            r2 = backtrack()
            for (j, old, new) in saved:
                domains[j] = old
            assignment[best_i] = None
            if r2 == 'timeout':
                return 'timeout'
            if r2:
                return True
        return False

    ok = backtrack()
    if ok is True:
        return [(assignment[i], mods[i]) for i in range(n)]
    return None

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN

    all_mods = list(range(2, MAX_MOD + 1))
    e_odd = [m for m in all_mods if m % 2 == 0 and (m // 2) % 2 == 1]
    # e_odd: 2,6,10,...,126  (32 of them)

    best = []
    best_len = 0

    def update(pairs):
        nonlocal best, best_len
        if pairs and len(pairs) > best_len:
            best = pairs
            best_len = len(pairs)

    # Candidate subsets: drop 30 of the 32 e_odd, keep 2.
    # Try a handful of representative pairs.
    keep_pairs = [
        (2, 6), (2, 126), (6, 126), (2, 10), (126, 122), (6, 10),
    ]
    tried = set()
    for a, b in keep_pairs:
        if time.time() > deadline - 4.0:
            break
        keep = {a, b}
        mods = [m for m in all_mods if (m not in e_odd) or (m in keep)]
        key = frozenset(mods)
        if key in tried:
            continue
        tried.add(key)
        # sort by ascending modulus (small first) to help MRV/AC
        mods_sorted = sorted(mods)
        pairs = _solve_csp(mods_sorted, deadline - 2.0, 400000)
        if pairs:
            update(pairs)
            if best_len == 97:
                return [[r, m] for r, m in best]

    # Fallback: greedy with most-constrained-first on the safe subset (drop all but 2 e_odd)
    if best_len < 97 and time.time() < deadline - 1.0:
        keep = {2, 6}
        mods = [m for m in all_mods if (m not in e_odd) or (m in keep)]
        # order: most constrained first = largest gcd with others ~ we just use ascending
        random.seed(0)
        order = sorted(mods)
        pairs = []
        for m in order:
            if time.time() > deadline - 0.5:
                break
            found = False
            rc = list(range(m))
            random.shuffle(rc)
            for r in rc:
                ok = True
                for (r2, m2) in pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs.append((r, m))
                    found = True
                    break
        if pairs:
            update(pairs)

    if not best:
        best = [(0, 2)]

    return [[r, m] for r, m in best]
