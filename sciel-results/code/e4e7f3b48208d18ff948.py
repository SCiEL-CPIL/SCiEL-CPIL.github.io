import random
import sys
import time
import collections

def construct(k, g):
    # k=3, g=16 expected
    if k != 3:
        return _fallback_construct(k, g)
        
    target_girth = g
    start_time = time.time()
    time_limit = 115.0 # Leave margin for 120s limit
    
    # Known best is 936. Moore bound 766.
    # Try 936 first, then smaller, then larger.
    # If 936 fails, maybe 800 is possible? Unlikely.
    # Let's try 936, 1000, 1200.
    
    n_candidates = [936, 950, 1000, 1100, 1200, 1500, 2000]
    
    for N in n_candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this N
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random 3-regular graph
            graph = _generate_random_3_regular(N)
            if graph is None:
                continue
                
            # Initial girth
            current_girth = _get_girth(graph, N, target_girth)
            
            if current_girth >= target_girth:
                return N, _extract_edges(graph)
                
            # Local search
            improved = True
            while improved and (time.time() - start_time < time_limit):
                improved = False
                best_girth = current_girth
                best_graph = None
                
                # Try a batch of random 2-switches
                # We want to find a switch that increases girth.
                
                # Pick a random edge to try to break
                # We don't know the shortest cycle, so we just pick random edges.
                # If we pick an edge on a short cycle, we might break it.
                
                for _ in range(100):
                    # Pick a random edge (u, v)
                    u = random.randint(0, N-1)
                    if not graph[u]: continue
                    v = random.choice(list(graph[u]))
                    
                    # Pick a random edge (x, y)
                    x = random.randint(0, N-1)
                    if not graph[x]: continue
                    y = random.choice(list(graph[x]))
                    
                    # Ensure (u,v) and (x,y) are distinct and valid for switch
                    # Switch replaces (u,v) and (x,y) with (u,x) and (v,y)
                    # Conditions:
                    # 1. u != x, u != y, v != x, v != y (no loops)
                    # 2. (u,x) and (v,y) not already edges (no multi-edges)
                    # 3. (u,v) != (x,y)
                    
                    if u == x or u == y or v == x or v == y:
                        continue
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                        
                    if _edge_exists(graph, u, x) or _edge_exists(graph, v, y):
                        continue
                        
                    # Perform switch
                    _remove_edge(graph, u, v)
                    _remove_edge(graph, x, y)
                    _add_edge(graph, u, x)
                    _add_edge(graph, v, y)
                    
                    new_girth = _get_girth(graph, N, target_girth)
                    
                    if new_girth > best_girth:
                        best_girth = new_girth
                        # Deep copy graph
                        best_graph = [set(neighbors) for neighbors in graph]
                        improved = True
                        if new_girth >= target_girth:
                            return N, _extract_edges(graph)
                    else:
                        # Revert
                        _remove_edge(graph, u, x)
                        _remove_edge(graph, v, y)
                        _add_edge(graph, u, v)
                        _add_edge(graph, x, y)
                        
                if improved:
                    current_girth = best_girth
                    graph = best_graph
                else:
                    # Stuck. Break and try new restart.
                    break

    return _fallback_construct(k, g)

def _generate_random_3_regular(N):
    # Configuration model
    stubs = []
    for i in range(N):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair them up
    graph = [set() for _ in range(N)]
    
    for i in range(0, 3*N, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if v in graph[u]:
            return None # Multi-edge
        graph[u].add(v)
        graph[v].add(u)
        
    return graph

def _edge_exists(graph, u, v):
    return v in graph[u]

def _add_edge(graph, u, v):
    graph[u].add(v)
    graph[v].add(u)

def _remove_edge(graph, u, v):
    graph[u].discard(v)
    graph[v].discard(u)

def _get_girth(graph, N, target_girth):
    # Returns girth, or target_girth if no cycle < target_girth found
    max_depth = target_girth // 2
    min_cycle = target_girth
    
    visited = [0] * N
    parent = [-1] * N
    dist = [0] * N
    timestamp = 0
    
    for start in range(N):
        timestamp += 1
        visited[start] = timestamp
        parent[start] = -1
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
                
            for v in graph[u]:
                if visited[v] == timestamp:
                    # v is in the current BFS tree
                    if v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < target_girth:
                                # We found a short cycle. 
                                # We can return early? 
                                # No, we want the *minimum* cycle length to know how much we improved.
                                # But if we just want to know if girth >= target_girth, 
                                # we can return min_cycle immediately if it's < target_girth?
                                # Actually, if we return early, we might miss a *shorter* cycle?
                                # No, BFS finds shortest paths. The first cycle found might not be the shortest.
                                # But we update min_cycle.
                                # To be safe, we continue to find the true minimum.
                                pass
                else:
                    visited[v] = timestamp
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    
    return min_cycle

def _extract_edges(graph):
    edges = []
    for i in range(len(graph)):
        for j in graph[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _fallback_construct(k, g):
    # Use the seed logic if we fail
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
