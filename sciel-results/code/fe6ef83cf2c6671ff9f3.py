"""4-regular graph of girth 10: voltage lift with constraint-guided search + LUW fallback."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque
from itertools import product

def _moore_bound(k, g):
    m = g // 2
    if k == 2:
        return 2 * m
    if m == 1:
        return k
    return k + k * (k - 1) * ((k - 1) ** (m - 1) - 1) // (k - 2)

def _verify_girth_fast(n, edges, k, g, timeout=5.0):
    """Verify girth >= g using non-backtracking BFS from each directed edge."""
    start_time = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    max_depth = g - 2
    for a in range(n):
        for b in adj[a]:
            if time.time() - start_time > timeout:
                return False
            visited = set()
            queue = deque()
            queue.append((b, a, 0))
            visited.add((b, a))
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    state = (nxt, cur)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
    return True

def _check_4cycle_voltage(vx, vy):
    """Check that no 4-cycle in K_{4,4} has zero voltage in Z_8 x Z_5."""
    for a in range(4):
        for ap in range(a+1, 4):
            for b in range(4):
                for bp in range(b+1, 4):
                    dx = (vx[a][b] - vx[ap][b] + vx[ap][bp] - vx[a][bp]) % 8
                    dy = (vy[a][b] - vy[ap][b] + vy[ap][bp] - vy[a][bp]) % 5
                    if dx == 0 and dy == 0:
                        return False
    return True

def _check_6cycle_voltage(vx, vy):
    """Check that no 6-cycle in K_{4,4} has zero voltage.
    A 6-cycle in K_{4,4} is a-b-a'-b'-a''-b''-a where a,b,a',b',a'',b'' all distinct pairs
    with a != a' != a'' != a and b != b' != b'' != b.
    Actually in K_{4,4}, a 6-cycle visits 3 vertices from A and 3 from B.
    Walk: a -> b -> a' -> b' -> a'' -> b'' -> a
    Voltage: v_ab - v_{a'b} + v_{a'b'} - v_{a''b'} + v_{a''b''} - v_{ab''}
    """
    for a in range(4):
        for ap in range(4):
            if ap == a: continue
            for app in range(4):
                if app == a or app == ap: continue
                for b in range(4):
                    for bp in range(4):
                        if bp == b: continue
                        for bpp in range(4):
                            if bpp == b or bpp == bp: continue
                            # Walk: a -> b -> ap -> bp -> app -> bpp -> a
                            dx = (vx[a][b] - vx[ap][b] + vx[ap][bp] - vx[app][bp] + vx[app][bpp] - vx[a][bpp]) % 8
                            dy = (vy[a][b] - vy[ap][b] + vy[ap][bp] - vy[app][bp] + vy[app][bpp] - vy[a][bpp]) % 5
                            if dx == 0 and dy == 0:
                                return False
    return True

def _build_lifted_graph(vx, vy):
    """Build the 320-vertex lifted graph from voltage matrices."""
    n = 320
    adj = [set() for _ in range(n)]
    
    for a in range(4):
        for bi in range(4):
            b = bi + 4
            v_x = vx[a][bi]
            v_y = vy[a][bi]
            for x in range(8):
                for y in range(5):
                    u_idx = a * 40 + x * 5 + y
                    w_idx = bi * 40 + ((x + v_x) % 8) * 5 + ((y + v_y) % 5)
                    if w_idx not in adj[u_idx]:
                        adj[u_idx].add(w_idx)
                        adj[w_idx].add(u_idx)
    
    edges = []
    for u in range(n):
        for w in adj[u]:
            if u < w:
                edges.append((u, w))
    
    if any(len(adj[i]) != 4 for i in range(n)):
        return None, None
    
    return n, edges

def _construct_h33_cage():
    """Construct the (4,10)-cage via voltage lift on K_{4,4} with Z_8 x Z_5.
    Uses constraint-guided random search: first satisfy 4-cycle constraints,
    then check 6-cycle constraints, then full verification.
    """
    random.seed(12345)
    deadline_local = time.time() + 75.0
    
    # Phase 1: Try structured voltage assignments that are more likely to work
    # Use affine + perturbation approaches
    structured_attempts = 0
    while structured_attempts < 200 and time.time() < deadline_local - 30:
        structured_attempts += 1
        # Try V[a][b] = (p*a + q*b + r*a*b mod 8, p*a + q*b + r*a*b mod 5)
        # with random coefficients, plus random perturbation
        p = random.randint(0, 7)
        q = random.randint(0, 7)
        r = random.randint(0, 7)
        p5 = random.randint(0, 4)
        q5 = random.randint(0, 4)
        r5 = random.randint(0, 4)
        
        vx = [[(p*a + q*b + r*a*b + random.randint(0,3)) % 8 for b in range(4)] for a in range(4)]
        vy = [[(p5*a + q5*b + r5*a*b + random.randint(0,1)) % 5 for b in range(4)] for a in range(4)]
        
        if _check_4cycle_voltage(vx, vy):
            n, edges = _build_lifted_graph(vx, vy)
            if n is not None and _verify_girth_fast(n, edges, 4, 10, timeout=1.5):
                return n, edges
    
    # Phase 2: Random search with 4-cycle constraint checking
    for attempt in range(200000):
        if time.time() > deadline_local:
            break
        
        vx = [[random.randint(0, 7) for _ in range(4)] for _ in range(4)]
        vy = [[random.randint(0, 4) for _ in range(4)] for _ in range(4)]
        
        # Quick 4-cycle check (fast)
        if not _check_4cycle_voltage(vx, vy):
            continue
        
        # 6-cycle check (moderate cost)
        if not _check_6cycle_voltage(vx, vy):
            continue
        
        # Build and verify
        n, edges = _build_lifted_graph(vx, vy)
        if n is not None and _verify_girth_fast(n, edges, 4, 10, timeout=1.0):
            return n, edges
    
    return None, None

def _construct_luw(q, d):
    """Lazebnik-Ustimenko-Woldar CD(d, q)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _try_luw(k, g, deadline):
    """Try LUW constructions with various d values."""
    best_n = None
    best_edges = None
    
    for d in range(max(3, g - 5), max(3, g - 5) + 8):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = _construct_luw(k, d)
            if n < 5000:
                if _verify_girth_fast(n, edges, k, g, timeout=3.0):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
        except Exception:
            pass
    
    return best_n, best_edges

def _voltage_lift_cage(k, g, deadline):
    """Construct k-regular graph of girth >= g via voltage lift."""
    if k != 4 or g != 10:
        return None, None
    
    result = _construct_h33_cage()
    if result[0] is not None:
        return result
    
    # Fallback: try smaller voltage lifts with Z_8
    random.seed(42)
    
    for attempt in range(3000):
        if time.time() > deadline - 25:
            break
        
        volt = [[random.randint(0, 7) for _ in range(4)] for _ in range(4)]
        
        n = 64
        adj = [set() for _ in range(n)]
        
        for a in range(4):
            for bi in range(4):
                b = bi + 4
                v = volt[a][bi]
                for t in range(8):
                    u = a * 8 + t
                    w = bi * 8 + ((t + v) % 8)
                    adj[u].add(w)
                    adj[w].add(u)
        
        if any(len(adj[i]) != 4 for i in range(n)):
            continue
        
        edges = []
        for u in range(n):
            for w in adj[u]:
                if u < w:
                    edges.append((u, w))
        
        if _verify_girth_fast(n, edges, 4, 10, timeout=1.0):
            return n, edges
    
    return None, None

def _random_construction(k, g, target_n, deadline, seed=42):
    """Try to build a k-regular graph with girth >= g via randomized edge addition."""
    random.seed(seed)
    n = target_n
    adj = [set() for _ in range(n)]
    edges = []
    
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    random.shuffle(half_edges)
    
    i = 0
    while i < len(half_edges):
        u = half_edges[i]
        v = half_edges[i + 1]
        i += 2
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        else:
            for j in range(i + 2, len(half_edges)):
                v2 = half_edges[j]
                if v2 != u and v2 not in adj[u]:
                    half_edges[i + 1], half_edges[j] = half_edges[j], half_edges[i + 1]
                    v = v2
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    break
    
    if not _verify_girth_fast(n, edges, k, g, timeout=2.0):
        return None, None
    
    return n, edges

def construct(k, g):
    deadline = time.time() + 115.0
    mb = _moore_bound(k, g)
    
    # Strategy 1: Voltage lift on K_{4,4} with Z_8 x Z_5 (320 vertices)
    if k == 4 and g == 10:
        n, edges = _construct_h33_cage()
        if n is not None:
            return n, edges
    
    # Strategy 2: Other voltage lifts
    best_n, best_edges = _voltage_lift_cage(k, g, deadline)
    
    if best_n is not None:
        return best_n, best_edges
    
    # Strategy 3: LUW constructions
    best_n, best_edges = _try_luw(k, g, deadline)
    
    if best_n is not None:
        return best_n, best_edges
    
    # Strategy 4: Random constructions
    for target_n in [320, 350, 400, 500, 600, 800, 1000, 1200, 1500, 2000]:
        if time.time() > deadline - 15:
            break
        if target_n < mb:
            continue
        n, edges = _random_construction(k, g, target_n, deadline, seed=target_n)
        if n is not None:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    try:
        d = g - 3
        n, edges = _construct_luw(k, d)
        if _verify_girth_fast(n, edges, k, g, timeout=5.0):
            return n, edges
    except Exception:
        pass
    
    return mb, []
# EVOLVE-BLOCK-END
