import random
import time
import heapq
from collections import deque, defaultdict

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    start_time = time.time()
    time_limit = 115  # leave 5s buffer
    
    # Moore bound
    half_g = (g - 1) // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(half_g + 1))
    
    # LUW as fallback
    try:
        luw_n, luw_edges = _luw_construct(k, g)
    except Exception:
        luw_n, luw_edges = None, None
    
    best_n = luw_n
    best_edges = luw_edges
    
    # Try local search from Moore bound upward
    for n in range(moore_bound, min(moore_bound + 300, (best_n or 10**9))):
        if time.time() - start_time > time_limit:
            break
        if best_n is not None and n >= best_n:
            break
        
        success = False
        for attempt in range(40):
            if time.time() - start_time > time_limit:
                break
            
            edges = _local_search(k, g, n, seed=attempt * 7919 + 42, time_limit=2.5)
            if edges is not None:
                best_n = n
                best_edges = edges
                success = True
                break
        
        if success:
            # Try to go even smaller? No, we're going upward from Moore bound
            # So this is already the smallest found
            pass
    
    if best_n is None:
        best_n, best_edges = _luw_construct(k, g)
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _local_search(k, g, n, seed=0, time_limit=3.0):
    """
    Generate a random k-regular graph and use 2-switch local search to eliminate short cycles.
    Returns list of edges if successful, None otherwise.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Generate random k-regular graph via configuration model
    adj, edge_set = _generate_random_k_regular(k, n, rng)
    if adj is None:
        return None
    
    # Local search: iteratively find and break short cycles using 2-switches
    max_iterations = 100000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            return None
        
        # Find a short cycle (length < g)
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            # No short cycles found - verify and return
            if _verify_girth_adj(adj, k, g, n):
                edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                return edges
            break
        
        # Try to break this cycle using 2-switches
        if not _break_cycle_2switch(adj, edge_set, cycle, n, g, rng):
            # If we can't break this specific cycle, try a random 2-switch
            if not _random_2switch(adj, edge_set, n, rng):
                return None
    
    return None


def _generate_random_k_regular(k, n, rng):
    """Generate a random k-regular graph using configuration model with retry."""
    if n * k % 2 != 0 or n < k:
        return None, None
    
    for attempt in range(100):
        stubs = [v for v in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            edge_set = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_set.add((u, v))
            return [list(s) for s in adj], edge_set
    
    return None, None


def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using BFS. Returns list of vertices or None."""
    half = (g - 1) // 2
    
    for start in range(n):
        parent = [-1] * n
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != -1:
                            path1.append(cur)
                            cur = parent[cur]
                        path2 = []
                        cur = v
                        while cur != -1:
                            path2.append(cur)
                            cur = parent[cur]
                        set1 = set(path1)
                        lca = None
                        for node in path2:
                            if node in set1:
                                lca = node
                                break
                        if lca is not None:
                            c1 = []
                            cur = u
                            while cur != lca:
                                c1.append(cur)
                                cur = parent[cur]
                            c1.append(lca)
                            c2 = []
                            cur = v
                            while cur != lca:
                                c2.append(cur)
                                cur = parent[cur]
                            cycle = c1 + c2[::-1]
                            if 3 <= len(cycle) < g:
                                return cycle
    return None


def _break_cycle_2switch(adj, edge_set, cycle, n, g, rng):
    """Try to break a cycle using 2-switches. Returns True if a switch was applied."""
    m = len(cycle)
    if m < 3:
        return True
    
    tried = set()
    candidates = []
    
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            
            u1, v1 = cycle[i], cycle[(i + 1) % m]
            u2, v2 = cycle[j], cycle[(j + 1) % m]
            
            for (a, b, c, d) in [(u1, v1, u2, v2), (u1, v1, v2, u2)]:
                if a == c or b == d:
                    continue
                new_e1 = (min(a, c), max(a, c))
                new_e2 = (min(b, d), max(b, d))
                if new_e1 in edge_set or new_e2 in edge_set:
                    continue
                if new_e1 == new_e2:
                    continue
                
                key = (new_e1, new_e2)
                if key in tried:
                    continue
                tried.add(key)
                candidates.append((u1, v1, u2, v2, a, b, c, d, new_e1, new_e2))
    
    if not candidates:
        return False
    
    # Pick a random candidate and apply it
    idx = rng.randrange(len(candidates))
    u1, v1, u2, v2, a, b, c, d, new_e1, new_e2 = candidates[idx]
    
    old_e1 = (min(u1, v1), max(u1, v1))
    old_e2 = (min(u2, v2), max(u2, v2))
    
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    edge_set.discard(old_e1)
    edge_set.discard(old_e2)
    
    adj[a].append(c)
    adj[c].append(a)
    adj[b].append(d)
    adj[d].append(b)
    edge_set.add(new_e1)
    edge_set.add(new_e2)
    
    return True


def _random_2switch(adj, edge_set, n, rng):
    """Perform a random 2-switch. Returns True if successful."""
    all_edges = list(edge_set)
    if len(all_edges) < 2:
        return False
    
    for _ in range(50):
        e1, e2 = rng.sample(all_edges, 2)
        u1, v1 = e1
        u2, v2 = e2
        
        for (a, b, c, d) in [(u1, v1, u2, v2), (u1, v1, v2, u2)]:
            if a == c or b == d:
                continue
            new_e1 = (min(a, c), max(a, c))
            new_e2 = (min(b, d), max(b, d))
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
            if new_e1 == new_e2:
                continue
            
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            edge_set.discard(e1)
            edge_set.discard(e2)
            
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            return True
    
    return False


def _verify_girth_adj(adj, k, g, n):
    """Verify k-regularity and girth >= g using adjacency list."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    half = (g - 1) // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True
