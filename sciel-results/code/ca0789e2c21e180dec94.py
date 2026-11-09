import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
_DEADLINE = 150.0

def _build_bad(n, colors):
    """For each colour c, a set of frozenset({u,v}) of fixed edges whose other
    endpoints would form a C6 if both were joined to the next vertex in colour c."""
    byc = defaultdict(set)
    for c in range(4):
        for (i, j) in byc[c]:
            for (a, b) in byc[c]:
                if i == a and j == b:
                    continue
                if i == b and j == a:
                    continue
                if i == a or i == b or j == a or j == b:
                    continue
                # endpoints {i,j} and {a,b} all distinct
                rem = {i, j} ^ {a, b}
                if len(rem) == 2:
                    x, y = rem
                    if x != y:
                        # joining (x,k) and (y,k) in colour c closes a C6
                        byc[c].add(frozenset((x, y)))
    return byc

def _search(n, t_end):
    """Backtracking: assign colours to edges (i,j) for i<j, vertex-by-vertex.
    Returns a valid colour list or None."""
    bad = _build_bad(n, [])
    edge_c = [None] * (n * (n - 1) // 2)
    E = n * (n - 1) // 2
    rnd = random.Random(12345)
    cols = [0, 1, 2, 3]
    rnd.shuffle(cols)

    def bt(k):
        if time.time() > t_end:
            return False
        if k == n:
            return True
        # edges (k, x) for x < k, in a random order
        xs = list(range(k))
        rnd.shuffle(xs)
        # order edges by fewest options first
        def opt_count(x):
            cnt = 0
            for c in cols:
                p = (k ^ x) ^ k ^ x  # placeholder
                # partner for (k,x) in colour c is the other endpoint of bad pair (x,partner)
                # we just check membership below; count via loop
                cnt += 1
            return cnt
        # cheaper ordering: sort by k-x (outer first) then shuffle handled
        for x in xs:
            ok = False
            for c in cols:
                if time.time() > t_end:
                    return False
                # partner: the bad pair containing x in colour c
                found = False
                for bp in bad[c]:
                    if x in bp:
                        partner = bp - {x}
                        partner = next(iter(partner))
                        # if (k,partner) already set to c -> C6
                        if edge_c[k * (k - 1) // 2 + (k - 1 - partner)] == c:
                            found = True
                            break
                if found:
                    continue
                # also (k,x) itself must not create C6 with two prior edges both in c
                # that is exactly the bad-pair rule above (one prior edge + (k,partner))
                edge_c[k * (k - 1) // 2 + (k - 1 - x)] = c
                byc = bad
                byc[c].add(frozenset((k, x)))
                ok = bt(k + 1)
                if ok:
                    return True
                byc[c].discard(frozenset((k, x)))
                edge_c[k * (k - 1) // 2 + (k - 1 - x)] = None
            if not ok:
                return False
        return False

    if bt(0):
        return edge_c
    return None

def _verify(n, colors):
    """Independent check: no colour class contains a C6."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c].add((i, j))
    # build adjacency lists per colour
    for c in range(4):
        g = [[] for _ in range(n)]
        for (i, j) in adj[c]:
            g[i].append(j)
            g[j].append(i)
        # check any 6-cycle: for all a<b<c<d<e<f, edges a-b,b-c,c-d,d-e,e-f,f-a in colour c
        # cheaper: DFS from each start with depth up to 6
        for s in range(n):
            # simple cycle detection of length 6 containing s
            stack = [(s, s, 0, {s})]
            found = False
            while stack and not found:
                u, start, d, vis = stack.pop()
                if d == 5:
                    if start in g[u]:
                        found = True
                        break
                    continue
                for v in g[u]:
                    if v == start:
                        if d == 5:
                            found = True
                            break
                        continue
                    if v in vis:
                        continue
                    stack.append((v, start, d + 1, vis | {v}))
            if found:
                return False
    return True

def construct():
    t0 = time.time()
    t_end = t0 + _DEADLINE - 5.0  # safety margin
    rnd = random.Random(7)

    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    # try to improve n, from high to low, with a time budget per attempt
    attempts = [(18, 20.0), (17, 90.0), (16, 20.0)]
    for n, budget in attempts:
        if time.time() > t_end:
            break
        a0 = time.time()
        while time.time() < min(a0 + budget, t_end):
            res = _search(n, min(a0 + budget, t_end))
            if res is not None:
                if _verify(n, res):
                    best_n = n
                    best_colors = res
                    break  # found a valid one for this n; go to next (higher already tried)
        # if found, continue to lower n only if we never found higher
        if best_n == n:
            break
        # otherwise keep trying next lower n in the list

    # final safety: if nothing beyond seed, return seed
    if best_n <= 9:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
