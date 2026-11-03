import random
import sys
import time
from collections import deque

def construct(k, g):
    # k=3, g=16 expected
    if k != 3:
        # Fallback for other k, though task specifies k=3
        return _fallback_construct(k, g)
        
    target_girth = g
    
    # Time budget
    start_time = time.time()
    time_limit = 110.0 # Leave some margin
    
    # Try N values starting from a reasonable lower bound
    # Moore bound for k=3, g=16 is around 766
    # We try a range of N values
    N_values = [800, 900, 1000, 1200, 1500, 2000, 3000, 5000, 10000]
    
    for N in N_values:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a graph of size N with girth >= target_girth
        # We do several random restarts with local search
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random 3-regular graph
            graph = _generate_random_3_regular(N)
            if graph is None:
                continue
                
            # Check if girth is already sufficient
            current_girth = _get_girth(graph, N, target_girth)
            
            if current_girth >= target_girth:
                return N, _extract_edges(graph)
                
            # Local search to increase girth
            improved = True
            while improved and (time.time() - start_time < time_limit):
                improved = False
                
                # Try multiple random 2-switches
                for _ in range(200):
                    if time.time() - start_time > time_limit:
                        break
                    
                    e1_idx, e2_idx = _pick_two_edges(graph, N)
                    if e1_idx is None:
                        continue
                    
                    u, v = e1_idx
                    x, y = e2_idx
                    
                    # Check if switch is valid (no loops, no multi-edges)
                    if u == x or u == y or v == x or v == y:
                        continue
                    if _edge_exists(graph, u, x) or _edge_exists(graph, v, y):
                        continue
                        
                    # Perform switch
                    _remove_edge(graph, u, v)
                    _remove_edge(graph, x, y)
                    _add_edge(graph, u, x)
                    _add_edge(graph, v, y)
                    
                    new_girth = _get_girth(graph, N, target_girth)
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
    edge_set = set()
    for i in range(0, 3*N, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None # Multi-edge
        edge_set.add(key)
        edges.append((u, v))
        
    # Build adjacency list
    graph = [[] for _ in range(N)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        
    return graph

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
    num_edges = 3 * N // 2
    # Pick random vertices and their neighbors
    for _ in range(20):
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

def _get_girth(graph, N, target_girth):
    # BFS from each vertex to find shortest cycle
    # Stop early if we find a cycle < target_girth
    # Return the minimum cycle length found, or target_girth if none found < target_girth
    min_cycle = target_girth
    half_target = target_girth // 2
    
    for start in range(N):
        # BFS
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = deque()
        queue.append(start)
        
        while queue:
            current = queue.popleft()
            depth = visited[current]
            if depth >= half_target:
                continue
                
            for neighbor in graph[current]:
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                else:
                    # Cycle found
                    if neighbor != parent[current]:
                        # The cycle length is at most depth + 1 + visited[neighbor]
                        # But we need to be more careful.
                        # If we find a back-edge, the cycle length is 
                        # dist(current) + dist(neighbor) + 1 - 2*dist(lca)
                        # However, a simpler check: if depth + 1 + visited[neighbor] < min_cycle,
                        # we have found a cycle of length at most that value.
                        # Actually, this is an upper bound.
                        # For exact girth, we'd need LCA, but for early stopping,
                        # if we find any cycle with upper bound < target_girth, we can stop.
                        
                        # A more precise check: if neighbor is at depth <= half_target - 1,
                        # and current is at depth <= half_target - 1, and they are connected,
                        # the cycle length is at most depth + 1 + visited[neighbor].
                        # If this is < target_girth, we have a short cycle.
                        
                        if depth + 1 + visited[neighbor] < min_cycle:
                            min_cycle = depth + 1 + visited[neighbor]
                            if min_cycle < target_girth:
                                return min_cycle
                                
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
