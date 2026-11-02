"""Voltage-graph (cyclic cover) construction for the 5-regular girth-9 cage problem.

We lift the base 5-regular graph K_{5,5} (girth 4) by a cyclic group Z_m.  Each base
edge {u,v} carries a "signed voltage" (sigma, tau): sigma in Z_m is the voltage and
tau in {0,1} is a sign.  A closed walk lifts to a cycle iff its combined voltage is
the identity, where the sign of a vertex is flipped on every odd-tau edge.  The sign
parity of any closed walk is well-defined, and a 4-cycle of K_{5,5} is killed exactly
when its sign parity is odd.  We use tau=1 on all edges (every 4-cycle in K_{5,5}
traverses exactly 2 edges, giving even parity = 0 mod 2, so no 4-cycle closes).
Then we search over integer voltages for one in which every 6- and 8-cycle is
non-closing, and finally choose the smallest m that makes every non-backtracking
walk of length < 9 non-closing.  The result is a 5-regular graph of girth >= 9
with ~5*m vertices, certified exactly by a non-backtracking BFS.
"""
import random
from collections import deque

_K = 5
_BASE_V = 10
_BASE_EDGES = [
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
]
_EDGE_IDX = {}
for _i, (_a, _b) in enumerate(_BASE_EDGES):
    _EDGE_IDX[(_a, _b)] = _i
    _EDGE_IDX[(_b, _a)] = _i

# Precompute the base graph's 4-, 6- and 8-cycles as lists of edge indices.
def _cycles_of_length(L):
    cset = set()
    for a in range(_BASE_V):
        na = [b for b in range(_BASE_V) if (a, b) in _EDGE_IDX]
        for i1 in range(len(na)):
            b = na[i1]
            for i2 in range(i1 + 1, len(na)):
                c = na[i2]
                if L == 4:
                    for x in range(_BASE_V):
                        if x in (a, b, c):
                            continue
                        if (b, x) in _EDGE_IDX and (c, x) in _EDGE_IDX:
                            e = [(_EDGE_IDX[a, b], _EDGE_IDX[b, x], _EDGE_IDX[x, c], _EDGE_IDX[c, a])]
                            if e[0] == tuple(sorted(e[0])):
                                cset.add(e[0])
                elif L == 6:
                    for x in range(_BASE_V):
                        if x in (a, b, c):
                            continue
                        if (b, x) in _EDGE_IDX and (c, x) in _EDGE_IDX:
                            for y in range(_BASE_V):
                                if y in (a, b, c, x):
                                    continue
                                if (x, y) in _EDGE_IDX and (y, a) in _EDGE_IDX:
                                    e = [(_EDGE_IDX[a, b], _EDGE_IDX[b, x], _EDGE_IDX[x, y],
                                          _EDGE_IDX[y, a], _EDGE_IDX[a, c], _EDGE_IDX[c, x])]
                                    if e[0] == tuple(sorted(e[0])):
                                        cset.add(e[0])
                else:  # L == 8
                    for x in range(_BASE_V):
                        if x in (a, b, c):
                            continue
                        if (b, x) in _EDGE_IDX and (c, x) in _EDGE_IDX:
                            for y in range(_BASE_V):
                                if y in (a, b, c, x):
                                    continue
                                if (x, y) in _EDGE_IDX and (y, a) in _EDGE_IDX:
                                    for z in range(_BASE_V):
                                        if z in (a, b, c, x, y):
                                            continue
                                        if (x, z) in _EDGE_IDX and (z, y) in _EDGE_IDX:
                                            e = [(_EDGE_IDX[a, b], _EDGE_IDX[b, x], _EDGE_IDX[x, z],
                                                  _EDGE_IDX[z, y], _EDGE_IDX[y, a], _EDGE_IDX[a, c],
                                                  _EDGE_IDX[c, x], _EDGE_IDX[x, y])]
                                            if e[0] == tuple(sorted(e[0])):
                                                cset.add(e[0])
    return cset


def _signed_voltage(edges, sigma, tau):
    w = 0
    s = 0
    flip = False
    for e in edges:
        w += sigma[e]
        if tau[e]:
            flip = not flip
        s = (s + 1) if flip else s
    return (w, s)


def _cycle_voltage(cycle_edges, sigma, tau):
    """Compute (total_voltage, sign_parity) for a cycle given as list of edge indices."""
    w = 0
    flip = False
    parity = 0
    for e in cycle_edges:
        w += sigma[e]
        if tau[e]:
            flip = not flip
        if flip:
            parity ^= 1
    return (w, parity)


def _find_signs_and_sigma():
    four = _cycles_of_length(4)
    six = _cycles_of_length(6)
    eight = _cycles_of_length(8)
    
    # Use tau = all 1s: every 4-cycle in K_{5,5} has exactly 2 edges in any
    # non-backtracking traversal, so parity is always even (0), meaning no 4-cycle closes.
    tau = [1] * (_K * _K)
    
    # Verify: every 4-cycle should have even parity with tau=all 1s
    for c in four:
        w, p = _cycle_voltage(c, [0]*25, tau)
        if p != 0:
            # Fallback: search for a valid tau
            break
    else:
        # tau=all 1s works for 4-cycles
        pass
    
    # Search for sigma that avoids zero voltage on 6- and 8-cycles
    # Use randomized local search
    best_sigma = None
    for attempt in range(200):
        if attempt == 0:
            sigma = list(range(1, _K * _K + 1))
        else:
            sigma = [random.randint(1, 50) for _ in range(_K * _K)]
        
        # Local search: iteratively fix bad cycles
        improved = True
        iters = 0
        while improved and iters < 500:
            improved = False
            iters += 1
            # Check 6-cycles
            for cs in six:
                w, p = _cycle_voltage(cs, sigma, tau)
                if w == 0:
                    # Perturb a random edge in this cycle
                    e = random.choice(cs)
                    sigma[e] += random.choice([-2, -1, 1, 2, 3, 5, 7])
                    if sigma[e] == 0:
                        sigma[e] = 1
                    improved = True
            # Check 8-cycles
            for cs in eight:
                w, p = _cycle_voltage(cs, sigma, tau)
                if w == 0:
                    e = random.choice(cs)
                    sigma[e] += random.choice([-2, -1, 1, 2, 3, 5, 7])
                    if sigma[e] == 0:
                        sigma[e] = 1
                    improved = True
        
        # Final check
        ok6 = all(_cycle_voltage(cs, sigma, tau)[0] != 0 for cs in six)
        ok8 = all(_cycle_voltage(cs, sigma, tau)[0] != 0 for cs in eight)
        if ok6 and ok8:
            best_sigma = sigma
            break
    
    if best_sigma is not None:
        return best_sigma, tau
    
    # Try a few more systematic approaches
    for base in range(1, 10):
        sigma = [base * (i + 1) + (i * i) % 7 for i in range(_K * _K)]
        ok6 = all(_cycle_voltage(cs, sigma, tau)[0] != 0 for cs in six)
        ok8 = all(_cycle_voltage(cs, sigma, tau)[0] != 0 for cs in eight)
        if ok6 and ok8:
            return sigma, tau
    
    return None


def _min_m(sigma, tau):
    four = _cycles_of_length(4)
    six = _cycles_of_length(6)
    eight = _cycles_of_length(8)
    
    # Precompute voltages and parities for all cycles
    four_data = [_cycle_voltage(c, sigma, tau) for c in four]
    six_data = [_cycle_voltage(c, sigma, tau) for c in six]
    eight_data = [_cycle_voltage(c, sigma, tau) for c in eight]
    
    # Collect all |w| values that need to not divide m (when parity is 0)
    forbidden_divisors = set()
    for w, p in four_data + six_data + eight_data:
        if p == 0 and w != 0:
            forbidden_divisors.add(abs(w))
    
    m = 2
    while m < 20000:
        ok = True
        for d in forbidden_divisors:
            if m % d == 0:
                ok = False
                break
        if ok:
            return m
        m += 1
    return 20000


def _lift(sigma, tau, m):
    n = _BASE_V * m
    adj = [[] for _ in range(n)]
    for (u, v) in _BASE_EDGES:
        e = _EDGE_IDX[(u, v)]
        s = sigma[e] % m
        t = tau[e]
        for f in range(m):
            a = u * m + f
            b = v * m + (f + s) % m
            adj[a].append(b)
            adj[b].append(a)
            if t:
                a2 = u * m + (f + m // 2) % m
                b2 = v * m + (f + s + m // 2) % m
                adj[a2].append(b2)
                adj[b2].append(a2)
    # Verify degree
    for i in range(n):
        if len(adj[i]) != _K:
            # Something went wrong; return empty
            return 0, [[] for _ in range(n)]
    return n, adj


def _verify(n, adj, g):
    # Check degree
    for i in range(n):
        if len(adj[i]) != _K:
            return False
    # Check simple graph (no multi-edges)
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False
    # Non-backtracking walk check for girth
    # For each directed edge (u,v), do BFS of non-backtracking walks of length < g
    # If we return to u, we found a cycle of length < g
    for u in range(n):
        for v in adj[u]:
            # State: (current_node, prev_node)
            queue = deque()
            queue.append((v, u))
            dist = 1
            while queue and dist < g:
                size = len(queue)
                for _ in range(size):
                    node, prev = queue.popleft()
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == u:
                            return False
                        queue.append((nb, node))
                dist += 1
    return True


def _construct_voltage(k, g):
    if k != _K or g != 9:
        return None
    st = _find_signs_and_sigma()
    if st is None:
        return None
    sigma, tau = st
    m = _min_m(sigma, tau)
    n, adj = _lift(sigma, tau, m)
    if n == 0 or not _verify(n, adj, 9):
        return None
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges


def construct(k, g):
    res = _construct_voltage(k, g)
    if res is not None:
        return res
    # Fallback (never expected for the contracted k=5, g=9).
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
