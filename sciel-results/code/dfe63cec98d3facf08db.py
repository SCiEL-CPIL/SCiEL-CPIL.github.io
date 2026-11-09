"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_graph(added_edges):
    """Build adjacency from base path + added edges."""
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    """Compute diameter using BFS from all vertices."""
    n = len(adj)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_triangle_free(adj):
    """Check if graph is triangle-free."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj_set[v]:
                    return False
    return True

def _is_valid(added_edges):
    """Check if added edges produce a valid graph."""
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = _build_graph(added_edges)
    if not _is_triangle_free(adj):
        return False
    diameter = _bfs_diameter(adj)
    if diameter < 0 or diameter > 4:
        return False
    return True

def _quick_valid(added_edges):
    """Faster validation: check triangle-free and diameter with early exit."""
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Triangle-free check
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj_set[v]:
                    return False
    
    # Diameter check with early exit
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] > 4:
                return False
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        if -1 in dist:
            return False
    return True

def _initial_solution():
    """Generate an initial valid solution."""
    return [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
            [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]

def _search_small_solutions(max_time=40):
    """Search for solutions with fewer than 10 edges."""
    start_time = time.time()
    n = 24
    
    # Precompute path edges
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u,v), max(u,v)))
    
    # All possible non-path edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in path_edges:
                all_edges.append((u, v))
    
    # Try 8-edge solutions: focus on hub-based structures
    # With 2 hubs, each can connect to ~6 vertices
    # Try pairs of hubs
    best = None
    
    # Strategy: try all pairs of "hub" vertices, then greedily add best connections
    for hub1 in range(2, n-2):
        if time.time() - start_time > max_time:
            break
        for hub2 in range(hub1+2, n-2):
            if abs(hub1 - hub2) < 3:  # hubs too close
                continue
            if time.time() - start_time > max_time:
                break
            
            # Determine which vertices each hub can connect to
            # A vertex v can connect to hub h if:
            # 1. (h,v) is not a path edge
            # 2. No triangle: for any neighbor w of v, (h,w) is not an edge
            #    (in the final graph)
            
            # For simplicity, first try: hub connects to all vertices at distance 2 from path
            # that don't create triangles
            
            # Let's build incrementally
            edges = []
            # Add hub1-hub2 edge if useful and valid
            if (hub1, hub2) not in path_edges:
                edges.append((hub1, hub2))
            
            # For each vertex, determine if it can connect to hub1 or hub2
            for v in range(n):
                if v == hub1 or v == hub2:
                    continue
                if time.time() - start_time > max_time:
                    break
                
                # Check if v can connect to hub1
                can_h1 = True
                if (hub1, v) in path_edges:
                    can_h1 = False
                else:
                    # Check triangle with existing edges
                    for (a, b) in edges:
                        if a == hub1 or b == hub1:
                            other = b if a == hub1 else a
                            if (v, other) in path_edges or (v, other) in [(min(x,y),max(x,y)) for x,y in edges]:
                                can_h1 = False
                                break
                        if a == v or b == v:
                            other = b if a == v else a
                            if (hub1, other) in path_edges:
                                can_h1 = False
                                break
                
                # Check if v can connect to hub2
                can_h2 = True
                if (hub2, v) in path_edges:
                    can_h2 = False
                else:
                    for (a, b) in edges:
                        if a == hub2 or b == hub2:
                            other = b if a == hub2 else a
                            if (v, other) in path_edges or (v, other) in [(min(x,y),max(x,y)) for x,y in edges]:
                                can_h2 = False
                                break
                        if a == v or b == v:
                            other = b if a == v else a
                            if (hub2, other) in path_edges:
                                can_h2 = False
                                break
                
                # Choose the hub that gives shorter path
                if can_h1 and can_h2:
                    # Pick closer one
                    d1 = abs(hub1 - v)
                    d2 = abs(hub2 - v)
                    if d1 <= d2:
                        edges.append((hub1, v))
                    else:
                        edges.append((hub2, v))
                elif can_h1:
                    edges.append((hub1, v))
                elif can_h2:
                    edges.append((hub2, v))
            
            # Remove hub-hub edge if not needed
            if len(edges) > 8:
                # Try without hub-hub
                edges_no_hh = [e for e in edges if e != (hub1, hub2)]
                if len(edges_no_hh) <= 9:
                    test_edges = [list(e) for e in edges_no_hh]
                    if _quick_valid(test_edges):
                        if best is None or len(test_edges) < len(best):
                            best = test_edges
            
            if len(edges) <= 9:
                test_edges = [list(e) for e in edges]
                if _quick_valid(test_edges):
                    if best is None or len(test_edges) < len(best):
                        best = test_edges
            
            if best is not None and len(best) <= 8:
                break
        if best is not None and len(best) <= 8:
            break
    
    return best

def _local_search(added_edges, max_time=20):
    """Improve solution via local search."""
    start_time = time.time()
    current = [list(e) for e in added_edges]
    if not _quick_valid(current):
        return added_edges
    
    best = current
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > max_time:
            break
        
        # Try removing edges
        for i in range(len(current)):
            if time.time() - start_time > max_time:
                break
            new_edges = current[:i] + current[i+1:]
            if _quick_valid(new_edges):
                current = new_edges
                improved = True
                if len(current) < len(best):
                    best = current
                break
        
        # Try single edge swaps
        if not improved and len(current) >= 2:
            n = 24
            path_edges = set()
            for u, v in P['edges']:
                path_edges.add((min(u,v), max(u,v)))
            
            current_set = set((min(u,v), max(u,v)) for u, v in current)
            
            for i in range(len(current)):
                if time.time() - start_time > max_time:
                    break
                u, v = current[i]
                # Try replacing edge i with a new edge
                for new_u in range(n):
                    if time.time() - start_time > max_time:
                        break
                    for new_v in range(new_u+1, n):
                        key = (new_u, new_v)
                        if key in path_edges or key in current_set:
                            continue
                        if key == (min(u,v), max(u,v)):
                            continue
                        
                        new_edges = current[:i] + [[new_u, new_v]] + current[i+1:]
                        if _quick_valid(new_edges):
                            current = new_edges
                            improved = True
                            if len(current) < len(best):
                                best = current
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    # Start with known 10-edge solution
    solution = _initial_solution()
    
    # Try to find smaller solutions
    if time.time() < deadline:
        small = _search_small_solutions(max_time=min(30, deadline - time.time()))
        if small is not None and len(small) < len(solution):
            solution = small
    
    # Local search to improve further
    if time.time() < deadline:
        remaining = min(15, deadline - time.time())
        if remaining > 0:
            solution = _local_search(solution, max_time=remaining)
    
    # Verify final solution
    if not _quick_valid(solution):
        solution = _initial_solution()
    
    # Ensure return type is correct
    result = []
    for e in solution:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
