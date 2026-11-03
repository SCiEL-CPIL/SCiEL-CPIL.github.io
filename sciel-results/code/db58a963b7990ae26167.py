"""Rewritten program: LUW q=4 construction + girth-preserving perfect matching."""
import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Lower bound on number of vertices for a k-regular graph of girth g."""
    if g % 2 == 1:
        r = (g - 3) // 2
        if k == 1:
            return 2
        total = 1 + k
        for i in range(1, r + 1):
            total += k * (k - 1) ** i
        return total
    else:
        r = g // 2 - 1
        if k == 1:
            return 2
        total = 0
        for i in range(r + 1):
            total += (k - 1) ** i
        return 2 * total


def _luw_q4(d, q):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=4 (GF(4))."""
    # GF(4) arithmetic using tables
    # Elements: 0, 1, a, b where a^2 = a+1, b = a^2 = a+1
    # Multiplication table:
    #   0: 0 0 0 0
    #   1: 0 1 a b
    #   a: 0 a a b
    #   b: 0 b b a
    # Let's encode: 0->0, 1->1, a->2, b->3
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 2, 3],
        [0, 3, 3, 2],
    ]
    # Addition in GF(4): characteristic 2
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    # Subtraction = addition (char 2)
    sub_table = add_table

    def add(x, y):
        return add_table[x][y]

    def sub(x, y):
        return sub_table[x][y]

    def mul(x, y):
        return mul_table[x][y]

    # Rule for CD(d, q)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The rule pattern repeats every 4 steps
        # For c = 5: r=0, c0=5 -> pattern index 0
        # For c = 6: r=1, c0=5 -> pattern index 1
        # For c = 7: r=2, c0=5 -> pattern index 2
        # For c = 8: r=3, c0=5 -> pattern index 3
        # Pattern: [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pattern[r]

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
    return len(order), edges, index, order


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_girth(n, edges, k, g):
    """Verify every vertex has degree k, graph is simple, and girth >= g."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found_short = False
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                break
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < g:
                        queue.append(v)
                else:
                    if dist[u] + dist[v] + 1 < g:
                        found_short = True
                        break
            if found_short:
                break
        if found_short:
            return False
    return True


def _hopcroft_karp(left_size, right_size, adj_left):
    """Hopcroft-Karp bipartite matching. adj_left[i] is list of right nodes connected to left i."""
    INF = float('inf')
    pair_u = [-1] * left_size
    pair_v = [-1] * right_size
    dist = [0] * left_size

    def bfs():
        queue = deque()
        for u in range(left_size):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_left[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in adj_left[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(left_size):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_u


def _compute_distances(adj, max_dist):
    """Compute distances up to max_dist from every vertex. Returns dist matrix."""
    n = len(adj)
    dist = [[max_dist + 1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            if dist[i][u] >= max_dist:
                continue
            for v in adj[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    if dist[i][v] < max_dist:
                        queue.append(v)
    return dist


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Moore bound
    mb = _moore_bound(k, g)

    # Strategy 1: LUW q=4 + perfect matching for k=5
    if k == 5 and g <= 10:
        # LUW q=4 with d=4 gives 4-regular graph with girth 9 (d+5=9 for odd d... wait d=4 is even, girth=d+6=10)
        # For q=4, d=4: girth = 4+6 = 10, n = 2*q^(d+1) = 2*4^5 = 2048? 
        # Actually n = 2 * q^(d+1) for CD(d,q)? Let me recalculate.
        # CD(d,q) has 2*q^(d+1) vertices? No, it has 2*q^d vertices on each side? 
        # Let me just construct and see.
        d = 4
        n_luw, edges_luw, index, order = _luw_q4(d, 4)
        
        if n_luw < 4000:  # reasonable size
            # Separate vertices into side 0 and side 1
            side0 = [i for i, v in enumerate(order) if v[0] == 0]
            side1 = [i for i, v in enumerate(order) if v[0] == 1]
            
            if len(side0) == len(side1):
                # Compute distances up to g-1
                adj_luw = _build_adj(n_luw, edges_luw)
                dist = _compute_distances(adj_luw, g - 1)
                
                # Build bipartite graph where (u, v) is an edge if dist[u][v] >= g
                # We need a perfect matching from side0 to side1
                left_size = len(side0)
                right_size = len(side1)
                
                # Map side indices to 0-based
                side0_idx = {v: i for i, v in enumerate(side0)}
                side1_idx = {v: i for i, v in enumerate(side1)}
                
                adj_left = [[] for _ in range(left_size)]
                for i, u in enumerate(side0):
                    for j, v in enumerate(side1):
                        if dist[u][v] >= g:
                            adj_left[i].append(j)
                
                # Check if a perfect matching exists
                matching_size, pair_u = _hopcroft_karp(left_size, right_size, adj_left)
                
                if matching_size == left_size:
                    # Perfect matching found! Add these edges
                    new_edges = list(edges_luw)
                    for i, u in enumerate(side0):
                        v = side1[pair_u[i]]
                        new_edges.append((u, v))
                    
                    # Verify
                    if _verify_girth(n_luw, new_edges, k, g):
                        return n_luw, new_edges
                
                # If no perfect matching with strict girth, try relaxing: allow dist >= g-1
                # and then verify
                adj_left2 = [[] for _ in range(left_size)]
                for i, u in enumerate(side0):
                    for j, v in enumerate(side1):
                        if dist[u][v] >= g - 1:
                            adj_left2[i].append(j)
                
                matching_size2, pair_u2 = _hopcroft_karp(left_size, right_size, adj_left2)
                if matching_size2 == left_size:
                    new_edges2 = list(edges_luw)
                    for i, u in enumerate(side0):
                        v = side1[pair_u2[i]]
                        new_edges2.append((u, v))
                    if _verify_girth(n_luw, new_edges2, k, g):
                        return n_luw, new_edges2

    # Strategy 2: LUW with q=k
    if k == 5:
        # For k=5, try LUW q=5 with appropriate d
        # girth of CD(d, q) is d+5 for odd d, d+6 for even d
        # We want girth >= g
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
        
        # LUW q=5 (prime field F5)
        q = 5
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = pattern[r]
        
        def neighbours_luw(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges_luw5 = []
        for vertex in order:
            for other in neighbours_luw(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw5.append((index[vertex], index[other]))
        n_luw5 = len(order)
        
        if _verify_girth(n_luw5, edges_luw5, k, g):
            return n_luw5, edges_luw5
    
    # Strategy 3: General LUW for other k values
    if k >= 2 and k <= 10:
        # Try LUW with q = smallest prime >= k
        def next_prime(n):
            if n <= 2:
                return 2
            if n % 2 == 0:
                n += 1
            else:
                n += 2
            while True:
                is_prime = True
                for i in range(3, int(math.sqrt(n)) + 1, 2):
                    if n % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    return n
                n += 2
        
        q = next_prime(k)
        if q > 50:
            q = k  # fallback to prime field if k is prime
        
        if q == k:
            d = g - 5
            if d % 2 == 0:
                d += 1
            d = max(3, d)
            
            rule = {}
            rule[2] = (1, 1)
            rule[3] = (2, 1)
            rule[4] = (1, 2)
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
                rule[c] = pattern[r]
            
            def neighbours_luw_gen(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = (x[c] + y[a] * x[b]) % q
                        else:
                            y[c] = (x[c] - x[a] * y[b]) % q
                    yield 1 - side, tuple(y)
            
            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges_gen = []
            for vertex in order:
                for other in neighbours_luw_gen(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges_gen.append((index[vertex], index[other]))
            n_gen = len(order)
            
            if _verify_girth(n_gen, edges_gen, k, g):
                return n_gen, edges_gen
    
    # Fallback: random search for small instances
    random.seed(42)
    if k <= 3 and g <= 8:
        for trial_n in range(mb, min(mb + 100, 200)):
            if time.time() > deadline:
                break
            for attempt in range(50):
                if time.time() > deadline:
                    break
                try:
                    edges = _random_regular_graph(trial_n, k)
                    if edges is not None:
                        if _verify_girth(trial_n, edges, k, g):
                            return trial_n, edges
                except Exception:
                    pass
    
    # Last resort: return a trivial construction if possible
    # For k=1: path graph
    if k == 1:
        n = max(mb, 2)
        edges = [(i, i + 1) for i in range(n - 1)]
        return n, edges
    
    # For general case, return a cycle if k=2
    if k == 2:
        n = max(mb, g)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    # Ultimate fallback: should not reach here for valid inputs
    return _moore_bound(k, g), []


def _random_regular_graph(n, k):
    """Generate a random k-regular simple graph on n vertices, or None if impossible."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    if k == 0:
        return []
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        edges.append((u, v))
    # Verify degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    return edges

# EVOLVE-BLOCK-END
