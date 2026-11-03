import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Attempt 1: Improved Voltage Lift Search
        res = _voltage_lift_search(start_time, deadline, g)
        if res is not None:
            return res
        
        # Attempt 2: Cayley Graph Search over Z_2 x Z_2 x Z_2 x Z_40
        res = _cayley_search(start_time, deadline, g)
        if res is not None:
            return res

        # Fallback: LUW construction
        return _luw_construct(k, g)

    # Generic fallback
    return _luw_construct(k, g)

def _voltage_lift_search(start_time, deadline, g):
    M = 32
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    E = len(base_edges)
    n = 10 * M
    eidx = [(u, v - 5) for (u, v) in base_edges]

    badj = [[] for _ in range(10)]
    for u, v in base_edges:
        badj[u].append(v)
        badj[v].append(u)

    def build(v):
        edges = []
        for idx in range(E):
            u, j = eidx[idx]
            w = v[u][j]
            vbase = 5 + j
            base_u = u * M
            base_v = vbase * M
            if w == 0:
                for x in range(M):
                    edges.append((base_u + x, base_v + x))
            else:
                for x in range(M):
                    y = x + w
                    if y >= M:
                        y -= M
                    edges.append((base_u + x, base_v + y))
        return edges

    def girth_ok(v, min_girth):
        max_depth = min_girth // 2
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for vv in badj[u]:
                    if vv == p:
                        continue
                    if u < 5 and vv >= 5:
                        w = v[u][vv - 5]
                    else:
                        w = M - v[vv][u - 5]
                    key = (vv, u)
                    if key in seen:
                        if d + 1 < min_girth:
                            return False
                    else:
                        seen.add(key)
                        q.append((vv, u, d + 1))
        return True

    def shortest_nb(v, min_girth):
        max_depth = min_girth // 2
        best = min_girth
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for vv in badj[u]:
                    if vv == p:
                        continue
                    key = (vv, u)
                    if key in seen:
                        L = d + 1
                        if L < best:
                            best = L
                    else:
                        seen.add(key)
                        q.append((vv, u, d + 1))
        return best

    def random_v():
        return [[random.randint(0, M - 1) for _ in range(5)] for _ in range(5)]

    def copy_v(v):
        return [row[:] for row in v]

    # 1) Deterministic candidates
    candidates = []
    candidates.append([[0] * 5 for _ in range(5)])
    for a in range(1, M):
        for b in range(1, M):
            if (a, b) in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 1)]:
                candidates.append([[((a * i + b * j) % M) for j in range(5)] for i in range(5)])
    for c in range(1, M):
        candidates.append([[c] * 5 for _ in range(5)])
    
    seen_cand = set()
    unique_cands = []
    for c in candidates:
        key = tuple(tuple(row) for row in c)
        if key not in seen_cand:
            seen_cand.add(key)
            unique_cands.append(c)
            
    for c in unique_cands:
        if time.time() > deadline - 3.0:
            break
        if girth_ok(c, g):
            return n, build(c)

    # 2) Randomized Local Search
    best_score = -1
    best_v = None
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        v = random_v()
        cur_score = shortest_nb(v, g)
        if cur_score >= g:
            return n, build(v)
        if cur_score > best_score:
            best_score = cur_score
            best_v = copy_v(v)
        
        cur_v = copy_v(v)
        cur_s = cur_score
        steps = 0
        max_steps = 60 # Increased steps
        while steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            idx = random.randrange(E)
            u, j = eidx[idx]
            old = cur_v[u][j]
            new = random.randint(0, M - 1)
            if new == old:
                continue
            cur_v[u][j] = new
            new_s = shortest_nb(cur_v, g)
            if new_s >= g:
                return n, build(cur_v)
            if new_s > cur_s:
                cur_s = new_s
                if cur_s > best_score:
                    best_score = cur_s
                    best_v = copy_v(cur_v)
            else:
                if random.random() < 0.7:
                    cur_v[u][j] = old
                else:
                    cur_s = new_s
                    
        if best_v is not None and best_score >= g:
            return n, build(best_v)

    if best_v is not None and best_score >= g:
        return n, build(best_v)
    return None

def _cayley_search(start_time, deadline, g):
    """Search for a Cayley graph over Z_2 x Z_2 x Z_2 x Z_40."""
    n = 320
    sizes = [2, 2, 2, 40]
    
    def encode(coords):
        r = 0
        for i, c in enumerate(coords):
            r = r * sizes[i] + c
        return r

    def decode(v):
        coords = []
        for s in sizes:
            coords.append(v % s)
            v //= s
        return tuple(coords)

    def build_edges(gens):
        edges = []
        seen = set()
        for v in range(n):
            c = decode(v)
            for gen in gens:
                nc = tuple((c[i] + gen[i]) % sizes[i] for i in range(4))
                u = encode(nc)
                if v < u:
                    edges.append((v, u))
                    seen.add((v, u))
        return edges

    def check_girth(edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != 4:
                return False

        # Girth check via BFS from each vertex (optimized: only check up to min_girth//2)
        max_depth = min_girth // 2
        for s in range(n):
            visited = {}
            q = deque()
            for w in adj[s]:
                visited[(w, s)] = 1
                q.append((w, s, 1))
            while q:
                node, parent, depth = q.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == s:
                        if depth + 1 < min_girth:
                            return False
                        continue
                    state = (neighbor, node)
                    if state in visited:
                        if depth + 1 < min_girth:
                            return False
                    else:
                        visited[state] = depth + 1
                        q.append((neighbor, node, depth + 1))
        return True

    # Search space
    # g1 = (a1, a2, a3, a4), g2 = (b1, b2, b3, b4)
    # a4, b4 in 1..19
    # a1..a3, b1..b3 in 0..1
    
    random.seed(42)
    
    # Try specific structured ones first
    # e.g. a4=1, b4=1
    for a4 in range(1, 20):
        if time.time() > deadline - 10:
            return None
        for b4 in range(a4, 20): # Avoid symmetry
            if time.time() > deadline - 10:
                return None
            # Randomize the Z_2 parts
            for _ in range(100):
                a1, a2, a3 = random.randint(0,1), random.randint(0,1), random.randint(0,1)
                b1, b2, b3 = random.randint(0,1), random.randint(0,1), random.randint(0,1)
                
                g1 = (a1, a2, a3, a4)
                g2 = (b1, b2, b3, b4)
                
                # Check inverses and duplicates
                # g1 != 0, g1 != -g1
                if all(x==0 for x in g1) or all(x==0 or (x==1 and sizes[i]==2) or (x==20 and sizes[i]==40) for i,x in enumerate(g1)):
                    continue
                if g1 == tuple((-x) % s for x, s in zip(g1, sizes)):
                    continue
                if g2 == tuple((-x) % s for x, s in zip(g2, sizes)):
                    continue
                if g1 == g2 or g1 == tuple((-x) % s for x, s in zip(g2, sizes)):
                    continue
                    
                gens = [g1, tuple((-x) % s for x, s in zip(g1, sizes)), 
                        g2, tuple((-x) % s for x, s in zip(g2, sizes))]
                
                edges = build_edges(gens)
                
                # Quick degree check
                if len(edges) != n * 2:
                    continue
                    
                if check_girth(edges, g):
                    return n, edges

    return None

def _luw_construct(k, g):
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
            yield (1 - side, tuple(y))

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
