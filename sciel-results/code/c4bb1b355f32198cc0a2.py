import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 4:
        return 0, []
        
    # Priority 1: Voltage graph construction
    result = _voltage_search(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    # Priority 2: Fallback to LUW if voltage search fails
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    return 0, []

def _voltage_search(k, g, deadline):
    """
    Search for a voltage graph lift with girth >= g.
    Tries base graphs K_{4,4} and Octahedron.
    """
    if k != 4:
        return None
        
    base_graphs = [
        ("K44", _get_k44_edges()),
        ("Oct", _get_octahedron_edges())
    ]
    
    best_result = None
    best_n = float('inf')
    
    for base_name, base_edges in base_graphs:
        base_n = len(set(v for e in base_edges for v in e))
        
        max_m = min(200, int(1000 / base_n) + 10)
        
        for m in range(10, max_m):
            if time.time() > deadline - 20:
                break
                
            n = m * base_n
            if n >= best_n:
                continue
                
            voltages = _find_voltages(base_edges, m, g, deadline - 10)
            if voltages is not None:
                lift_edges = _build_lift(base_edges, voltages, m)
                if _verify(n, lift_edges, k, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, lift_edges)
                        
    return best_result

def _get_k44_edges():
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges

def _get_octahedron_edges():
    opposites = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            if j != opposites[i]:
                edges.append((i, j))
    return edges

def _enumerate_walks(base_edges, max_len):
    """
    Enumerate all non-backtracking closed walks of length 3 to max_len-1.
    Returns a list of lists of (edge_idx, sign).
    """
    base_n = len(set(v for e in base_edges for v in e))
    adj = [[] for _ in range(base_n)]
    edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)
        
    walks = []
    
    for L in range(3, max_len):
        for u, v in base_edges:
            idx, sign = edge_map[(u, v)]
            path = [(idx, sign)]
            prev = u
            cur = v
            
            stack = [(prev, cur, tuple(path))]
            
            while stack:
                p, c, pth = stack.pop()
                if len(pth) == L:
                    if c == u:
                        walks.append(list(pth))
                    continue
                for (next_v, eidx) in adj[c]:
                    if next_v == p:
                        continue
                    e_u, e_v = base_edges[eidx]
                    if e_u == c and e_v == next_v:
                        s = 1
                    else:
                        s = -1
                    stack.append((c, next_v, pth + ((eidx, s),)))
            
    return walks

def _find_voltages(base_edges, m, g, deadline):
    """
    Find voltages in Z_m such that no closed non-backtracking walk of length < g sums to 0.
    Uses Simulated Annealing with incremental updates.
    """
    n_edges = len(base_edges)
    walks = _enumerate_walks(base_edges, g)
    
    if not walks:
        return [0] * n_edges
    
    # For each edge, precompute which walks use it and with what sign
    edge_to_walks = [[] for _ in range(n_edges)]
    for wi, walk in enumerate(walks):
        for eidx, sign in walk:
            edge_to_walks[eidx].append((wi, sign))
    
    # Precompute walk contributions: for each walk, list of (edge_idx, sign)
    walk_sigs = []
    for walk in walks:
        walk_sigs.append(walk)
    
    # Initial random voltages
    random.seed(42)
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    # Compute initial walk sums
    walk_sums = [0] * len(walks)
    for wi, walk in enumerate(walks):
        s = 0
        for eidx, sign in walk:
            s = (s + sign * voltages[eidx]) % m
        walk_sums[wi] = s
    
    current_violations = sum(1 for s in walk_sums if s == 0)
    if current_violations == 0:
        return voltages[:]
    
    # Simulated Annealing with incremental updates
    T = 10.0
    T_min = 0.01
    cooling = 0.97
    steps = 0
    max_steps = 20000
    restart_count = 0
    max_restarts = 10
    
    while T > T_min and time.time() < deadline:
        e_idx = random.randint(0, n_edges - 1)
        old_v = voltages[e_idx]
        new_v = random.randint(0, m - 1)
        if new_v == old_v:
            continue
            
        voltages[e_idx] = new_v
        
        # Incrementally update walk sums for walks using this edge
        delta_v = (new_v - old_v) % m
        new_violations = current_violations
        for wi, sign in edge_to_walks[e_idx]:
            old_sum = walk_sums[wi]
            was_zero = (old_sum == 0)
            new_sum = (old_sum + sign * delta_v) % m
            walk_sums[wi] = new_sum
            is_zero = (new_sum == 0)
            if was_zero and not is_zero:
                new_violations -= 1
            elif not was_zero and is_zero:
                new_violations += 1
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_violations = new_violations
        else:
            # Revert
            voltages[e_idx] = old_v
            for wi, sign in edge_to_walks[e_idx]:
                old_sum = walk_sums[wi]
                new_sum = (old_sum - sign * delta_v) % m
                walk_sums[wi] = new_sum
        
        T *= cooling
        steps += 1
        
        if current_violations == 0:
            return voltages[:]
        
        # Restart if stuck for too long
        if steps % 5000 == 0 and current_violations > 0:
            restart_count += 1
            if restart_count > max_restarts:
                break
            # Reinitialize with fresh random voltages
            random.seed(42 + restart_count * 1000)
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            walk_sums = [0] * len(walks)
            for wi, walk in enumerate(walks):
                s = 0
                for eidx, sign in walk:
                    s = (s + sign * voltages[eidx]) % m
                walk_sums[wi] = s
            current_violations = sum(1 for s in walk_sums if s == 0)
            T = 10.0
            steps = 0
            
    if current_violations == 0:
        return voltages[:]
    return None

def _build_lift(base_edges, voltages, m):
    base_n = len(set(v for e in base_edges for v in e))
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
                
    return sorted(set(edges))

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    return n, edges_list

def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
