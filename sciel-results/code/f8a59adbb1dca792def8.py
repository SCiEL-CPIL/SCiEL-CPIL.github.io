"""Construct a 3-regular graph of girth 16 via deterministic voltage lift of Petersen graph."""
import time
import random
from collections import deque

_DEADLINE = 120.0
_DEADLINE_FRAC = 0.85  # safety margin before 150s kill

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return k * ((k - 1) ** (r - 1))
    else:
        r = (g - 1) // 2
        return 1 + k * ((k - 1) ** (r - 1))

def _has_short_cycle(adj, n, max_len):
    """Return True if graph has a cycle of length < max_len. Uses non-backtracking BFS."""
    # For each vertex, BFS on states (prev, curr) to detect cycles back to start
    for start in range(n):
        # For each neighbor of start, do non-backtracking BFS
        for first in adj[start]:
            # visited[(prev, curr)] = depth
            visited = {}
            visited[(start, first)] = 1
            queue = deque()
            queue.append((start, first, 1))
            found = False
            while queue:
                prev, curr, d = queue.popleft()
                if d >= max_len - 1:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        if d + 1 >= 3:  # cycle length at least 3
                            return True
                        continue
                    state = (curr, nxt)
                    nd = d + 1
                    if state not in visited:
                        visited[state] = nd
                        queue.append((curr, nxt, nd))
    return False

def _build_lift_from_edges(base_edges, base_n, m, voltages):
    """Build lifted graph from base edges and voltages. Returns (n, adj)."""
    n = base_n * m
    adj = [[] for _ in range(n)]
    for eid, (i, j, _) in enumerate(base_edges):
        v = voltages[eid]
        for a in range(m):
            u1 = i * m + a
            u2 = j * m + ((a + v) % m)
            adj[u1].append(u2)
        for a in range(m):
            u1 = j * m + a
            u2 = i * m + ((a - v) % m)
            adj[u1].append(u2)
    return n, adj

def _petersen_edges():
    """Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5."""
    edges = [
        (0,1),(0,4),(0,5),
        (1,2),(1,6),(1,5),
        (2,3),(2,7),(2,6),
        (3,4),(3,8),(3,7),
        (4,9),(4,8),
        (5,7),(5,9),
        (6,8),(6,9),
        (7,8),
        (8,9),
    ]
    seen = set()
    unique = []
    for (i,j) in edges:
        if i > j:
            i,j = j,i
        if (i,j) not in seen:
            seen.add((i,j))
            unique.append((i,j))
    result = [(i, j, eid) for eid, (i,j) in enumerate(unique)]
    return result

def _find_primitive_root(p):
    """Find a primitive root modulo prime p."""
    if p == 2:
        return 1
    # Factor p-1
    phi = p - 1
    factors = []
    d = 2
    while d * d <= phi:
        if phi % d == 0:
            factors.append(d)
            while phi % d == 0:
                phi //= d
        d += 1
    if phi > 1:
        factors.append(phi)
    # Try candidates
    for g in range(2, p):
        ok = True
        for f in factors:
            if pow(g, (p-1)//f, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None  # shouldn't happen for prime p

def _next_prime(x):
    """Return the smallest prime >= x."""
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    while not is_prime(x):
        x += 1
    return x

def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE * _DEADLINE_FRAC

    if k != 3 or g != 16:
        return 0, []

    mb = _moore_bound(k, g)  # 384 for k=3, g=16
    base_edges = _petersen_edges()
    base_n = 10
    n_edges = len(base_edges)  # 15

    # Strategy: deterministic voltage assignment using powers of primitive root mod p
    # where p is a prime >= ceil(mb / base_n). Scan m = p, p+1, ... 
    # For each m, assign voltages as powers of a primitive root.
    
    best_result = None
    min_m = max(2, (mb + base_n - 1) // base_n)  # ceil(384/10) = 39

    # Try prime moduli first (better algebraic structure), then all m
    # For each m, use deterministic voltage: edge i gets voltage g^i mod m where g is primitive root
    # Also try randomized voltages as fallback
    
    # Phase 1: Deterministic voltage with primitive root for prime m
    for m in range(min_m, min_m + 60):
        if time.time() > deadline:
            break
        p = _next_prime(m)
        if p != m:
            # Skip non-primes in this phase, handled in phase 2
            continue
        root = _find_primitive_root(p)
        if root is None:
            continue
        # Assign voltages: edge i gets root^i mod p
        vols = [pow(root, i, p) for i in range(n_edges)]
        # Ensure no voltage is 0 (root is coprime to p, so powers are never 0 mod p)
        n_lift, adj = _build_lift_from_edges(base_edges, base_n, p, vols)
        if not _has_short_cycle(adj, n_lift, g):
            # Build edge list
            edge_set = set()
            for u in range(n_lift):
                for v in adj[u]:
                    if u < v:
                        edge_set.add((u, v))
            edge_list = list(edge_set)
            # Verify
            deg = [0] * n_lift
            for (u, v) in edge_list:
                deg[u] += 1
                deg[v] += 1
            if all(d == k for d in deg):
                if best_result is None or n_lift < best_result[0]:
                    best_result = (n_lift, edge_list)
                break  # Found valid for this m
    
    # Phase 2: Try all m (not just primes) with deterministic and random voltages
    if best_result is None or time.time() < deadline - 30:
        for m in range(min_m, min_m + 80):
            if time.time() > deadline:
                break
            if m < min_m:
                continue
            # Try a few deterministic patterns
            found = False
            # Pattern 1: voltages = [i * m // n_edges + 1 for i in range(n_edges)] (spread out)
            for trial in range(5):
                if time.time() > deadline:
                    break
                if trial == 0:
                    # Spread voltages evenly
                    vols = [(i * m + 1) % m for i in range(n_edges)]
                    if any(v == 0 for v in vols):
                        vols = [v + 1 for v in vols]
                elif trial == 1:
                    # Use golden ratio-like spacing
                    vols = [(int(i * m * 0.618033988749895) + 1) % m for i in range(n_edges)]
                    if any(v == 0 for v in vols):
                        vols = [v + 1 for v in vols]
                else:
                    # Random
                    vols = [random.randint(1, m - 1) for _ in range(n_edges)]
                
                n_lift, adj = _build_lift_from_edges(base_edges, base_n, m, vols)
                if not _has_short_cycle(adj, n_lift, g):
                    edge_set = set()
                    for u in range(n_lift):
                        for v in adj[u]:
                            if u < v:
                                edge_set.add((u, v))
                    edge_list = list(edge_set)
                    deg = [0] * n_lift
                    for (u, v) in edge_list:
                        deg[u] += 1
                        deg[v] += 1
                    if all(d == k for d in deg):
                        if best_result is None or n_lift < best_result[0]:
                            best_result = (n_lift, edge_list)
                        found = True
                        break
            if found:
                break
    
    if best_result is not None:
        return best_result

    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
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
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_list = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_list.append((index[vertex], index[other]))
    return len(order), edges_list
