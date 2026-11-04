"""Improved approach: random 3-regular graph generation with efficient girth checking and repair."""
import random
import time
from collections import deque

def _random_regular_graph(n, k, rng):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    rng.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edges:
            return None
        edges.add(key)
    return list(edges)

def _build_adj(edges, n):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_girth_fast(adj, n, girth):
    """Check if graph has girth >= girth using BFS from each vertex.
    Returns True if girth is at least girth, False otherwise."""
    # For each vertex, BFS up to depth (girth-1)//2
    # If we find a vertex at distance d1 via one path and d2 via another path
    # from the same start, and d1+d2 < girth, we have a short cycle.
    # More precisely: for each start vertex, do BFS and track distances.
    # A cycle of length L exists iff there's an edge (u,v) where dist(start,u) + dist(start,v) + 1 < girth
    # for some start, but that's not quite right either.
    
    # Better approach: for each vertex s, do BFS. If we encounter a vertex that 
    # was already visited at a shorter or equal distance, and the total path length
    # through s is < girth, we have a cycle.
    
    # Actually, the cleanest: for each edge (u,v), check if there's a path from u to v
    # of length < girth-1 not using (u,v). This is expensive.
    
    # Practical approach for small girth: for each vertex, do BFS up to depth (girth-1)//2.
    # Record the parent. If we reach a vertex that's an ancestor in the BFS tree,
    # the cycle length is depth + ancestor_depth + 1.
    
    half = (girth - 1) // 2  # For girth=16, half=7
    
    for s in range(n):
        # BFS from s up to depth half
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle: u-v edge where both are reached
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        return False
    return True

def _find_short_cycle(adj, n, girth):
    """Find a cycle of length < girth. Returns list of vertices or None."""
    half = (girth - 1) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != -1:
                            path1.append(cur)
                            cur = parent[cur]
                        path1.reverse()
                        path2 = []
                        cur = v
                        while cur != -1:
                            path2.append(cur)
                            cur = parent[cur]
                        path2.reverse()
                        # Find LCA
                        # Both paths go to s
                        # Cycle is path1 + path2 reversed (excluding s)
                        cycle = path1 + path2[:0:-1]
                        return cycle
    return None

def _edge_swap(adj, e1, e2, n):
    """Try to swap edges to break a cycle. Returns new adj or None."""
    u1, v1 = e1
    u2, v2 = e2
    for (a, b), (c, d) in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
        if a == b or c == d:
            continue
        if b in adj[a] or d in adj[c]:
            continue
        # Check the other direction too
        if a in adj[b] or c in adj[d]:
            continue
        adj2 = [set(s) for s in adj]
        adj2[u1].discard(v1)
        adj2[v1].discard(u1)
        adj2[u2].discard(v2)
        adj2[v2].discard(u2)
        adj2[a].add(b)
        adj2[b].add(a)
        adj2[c].add(d)
        adj2[d].add(c)
        if all(len(adj2[v]) == 3 for v in range(n)):
            return adj2
    return None

def _repair_short_cycles(adj, girth, n, rng, max_swaps=500):
    """Try to repair short cycles via edge swaps."""
    for _ in range(max_swaps):
        cycle = _find_short_cycle(adj, n, girth)
        if cycle is None:
            return adj
        
        # Get cycle edges
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            cycle_edges.add((min(u, v), max(u, v)))
        # Also add closing edge
        u, v = cycle[-1], cycle[0]
        cycle_edges.add((min(u, v), max(u, v)))
        
        ce = list(cycle_edges)
        rng.shuffle(ce)
        
        repaired = False
        for e1 in ce:
            u1, v1 = e1
            # Find candidate edges not sharing vertices with e1
            candidates = []
            for u2 in range(n):
                if u2 == u1 or u2 == v1:
                    continue
                for v2 in adj[u2]:
                    if v2 == u1 or v2 == v1:
                        continue
                    e2 = (min(u2, v2), max(u2, v2))
                    if e2 in cycle_edges:
                        continue
                    candidates.append(e2)
                    if len(candidates) > 20:
                        break
                if len(candidates) > 20:
                    break
            
            if not candidates:
                continue
            
            e2 = rng.choice(candidates)
            new_adj = _edge_swap(adj, e1, e2, n)
            if new_adj is not None:
                adj = new_adj
                repaired = True
                break
        
        if not repaired:
            return None
    
    return adj

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        return _construct_luw(k, g)
    
    girth = g
    rng = random.Random(42)
    start_time = time.time()
    
    best_n = None
    best_edges = None
    
    # Try sizes from Moore bound upward
    # Moore bound for (3,16): 2*(2^7 - 1) + 2 = 254... let's use 300 as start
    for base_size in range(300, 940, 2):
        if time.time() - start_time > 100:
            break
        n = base_size
        if n % 2 != 0:
            n += 1
        if n > 936:
            break
        
        for trial in range(30):
            if time.time() - start_time > 105:
                break
            edges = _random_regular_graph(n, 3, rng)
            if edges is None:
                continue
            
            adj = _build_adj(edges, n)
            
            if _check_girth_fast(adj, n, girth):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break
        
        if best_n is not None and best_n == n:
            # Found valid graph at this size, try smaller
            break
    
    # If we found something, try to improve with repair on slightly smaller graphs
    if best_n is None:
        # Try repair approach
        for base_size in range(400, 936, 2):
            if time.time() - start_time > 100:
                break
            n = base_size
            if n % 2 != 0:
                n += 1
            
            for trial in range(20):
                if time.time() - start_time > 105:
                    break
                edges = _random_regular_graph(n, 3, rng)
                if edges is None:
                    continue
                adj = _build_adj(edges, n)
                
                if not _check_girth_fast(adj, n, girth):
                    adj = _repair_short_cycles(adj, girth, n, rng, max_swaps=200)
                    if adj is None:
                        continue
                    if not _check_girth_fast(adj, n, girth):
                        continue
                
                # Valid
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = [(min(u, v), max(u, v)) for u in range(n) for v in adj[u] if u < v]
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_luw(k, g)

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
