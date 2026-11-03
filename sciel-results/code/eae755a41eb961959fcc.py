import random
import sys
import time

def construct(k, g):
    # k=3, g=16 expected
    if k != 3:
        # Fallback for other k, though task specifies k=3
        return _fallback_construct(k, g)
        
    target_girth = g
    # Moore bound for k=3, g=16 is 766
    # Start slightly above to ensure existence probability is decent
    # But we want to minimize N.
    
    # We will try increasing N until we find a graph with girth >= target_girth
    # and return it.
    
    # Time budget
    start_time = time.time()
    time_limit = 100.0 # Leave some margin
    
    # Try N values
    # The seed program likely produces N ~ 500,000.
    # We can easily beat that with N ~ 1000-10000.
    
    for N in [800, 1000, 1500, 2000, 3000, 5000, 10000, 20000]:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a graph of size N with girth >= target_girth
        # We do a few random restarts with local search
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random 3-regular graph
            graph = _generate_random_3_regular(N)
            if graph is None:
                continue
                
            # Local search to increase girth
            current_girth = _get_girth(graph, N)
            
            if current_girth >= target_girth:
                return N, _extract_edges(graph)
                
            # Try to improve
            improved = True
            while improved and (time.time() - start_time < time_limit):
                improved = False
                # Find a short cycle
                # For efficiency, just do random 2-switches and check girth
                # Or find shortest cycle and try to break it
                
                # Random 2-switch
                for _ in range(100):
                    e1_idx, e2_idx = _pick_two_edges(graph, N)
                    if e1_idx is None:
                        continue
                    
                    u, v = e1_idx
                    x, y = e2_idx
                    
                    # Check if switch is valid (no loops, no multi-edges)
                    # Edges are (u,v) and (x,y)
                    # New edges (u,x) and (v,y)
                    # Check if (u,x) or (v,y) already exist or are loops
                    if u == x or u == y or v == x or v == y:
                        continue
                    if _edge_exists(graph, u, x) or _edge_exists(graph, v, y):
                        continue
                        
                    # Perform switch
                    _remove_edge(graph, u, v)
                    _remove_edge(graph, x, y)
                    _add_edge(graph, u, x)
                    _add_edge(graph, v, y)
                    
                    new_girth = _get_girth(graph, N)
                    if new_girth > current_girth:
                        current_girth = new_girth
                        improved = True
                        if current_girth >= target_girth:
                            return N, _extract_edges(graph)
                        break # Restart search loop with new girth
                    else:
                        # Revert
                        _remove_edge(graph, u, x)
                        _remove_edge(graph, v, y)
                        _add_edge(graph, u, v)
                        _add_edge(graph, x, y)
                        
            if current_girth >= target_girth:
                return N, _extract_edges(graph)

    # If we didn't find one in the loop, return the best we can or a fallback
    # This should not happen if N is large enough
    return _fallback_construct(k, g)

def _generate_random_3_regular(N):
    # Configuration model
    # 3N stubs
    stubs = []
    for i in range(N):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair them up
    edges = []
    for i in range(0, 3*N, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if _edge_exists_in_list(edges, u, v):
            return None # Multi-edge
        edges.append((u, v))
        
    # Build adjacency list
    graph = [[] for _ in range(N)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        
    return graph

def _edge_exists_in_list(edges, u, v):
    for e in edges:
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            return True
    return False

def _edge_exists(graph, u, v):
    return v in graph[u]

def _add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)

def _remove_edge(graph, u, v):
    graph[u].remove(v)
    graph[v].remove(u)

def _pick_two_edges(graph, N):
    # Pick two random edges
    # Total edges = 3N/2
    num_edges = 3 * N // 2
    # We can just pick random vertices and their neighbors
    # To ensure distinct edges, pick random u, v in its neighbors
    # Then pick random x, y not forming same edge
    for _ in range(10):
        u = random.randint(0, N-1)
        if not graph[u]: continue
        v = random.choice(graph[u])
        
        x = random.randint(0, N-1)
        if not graph[x]: continue
        y = random.choice(graph[x])
        
        # Ensure (u,v) != (x,y)
        if (u == x and v == y) or (u == y and v == x):
            continue
            
        return (u, v), (x, y)
    return None, None

def _get_girth(graph, N):
    # BFS from each vertex to find shortest cycle
    # Since k=3, we can stop BFS at depth target_girth//2
    # But we don't know target_girth here, so we just find the global min
    # Optimization: Only need to check up to depth 8 for g=16
    min_cycle = 1000
    for start in range(N):
        # BFS
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            depth = visited[current]
            if depth >= 8: # Enough to detect cycles up to 16
                continue
                
            for neighbor in graph[current]:
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                else:
                    # Cycle found
                    # Length is depth + 1 + (distance from neighbor to start)
                    # Actually, if neighbor is visited, and neighbor != parent[current]
                    if neighbor != parent[current]:
                        # Find LCA? Or just calculate cycle length
                        # Simpler: if we hit a visited node, the cycle length is 
                        # depth(current) + 1 + depth(neighbor) - 2*depth(LCA)
                        # But we don't have LCA easily.
                        # However, for small depths, we can just trace back.
                        # Or simpler: if we find an edge to a visited node, 
                        # the cycle length is at most depth(current) + 1 + depth(neighbor)
                        # Actually, if we just want to know if there's a cycle < 16,
                        # we can check if depth(current) + 1 + depth(neighbor) < 16?
                        # No, that's an upper bound on the cycle length via that path.
                        # The actual cycle might be shorter.
                        # But if we find *any* cycle, we can compute its exact length.
                        
                        # To be safe and fast, if we find a back-edge, we can compute the cycle length
                        # by tracing back from neighbor to start? No, neighbor might be in a different branch.
                        
                        # Efficient method:
                        # When we see an edge (u, v) where v is visited and v != parent[u]:
                        # The cycle length is dist[u] + dist[v] + 1 - 2*dist[lca(u,v)]
                        # Since we are doing BFS, we can just check if the sum of distances is small.
                        # If dist[u] + dist[v] + 1 < 16, then there is a cycle of length <= that sum.
                        # So if dist[u] + dist[v] + 1 < 16, we found a short cycle.
                        
                        if depth + 1 + visited[neighbor] < min_cycle:
                            # This is an upper bound on the cycle length
                            # If this upper bound is < 16, we definitely have a short cycle
                            # But we want the *exact* girth to stop early?
                            # Actually, we just need to know if girth >= 16.
                            # So if we find a cycle of length < 16, we return it.
                            # If we finish BFS for all nodes and no cycle < 16 found, girth >= 16.
                            
                            # Let's just update min_cycle with this potential cycle length
                            # It might be an overestimate, but if it's < 16, it's valid.
                            # If it's >= 16, we ignore.
                            if depth + 1 + visited[neighbor] < min_cycle:
                                min_cycle = depth + 1 + visited[neighbor]
                                
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
