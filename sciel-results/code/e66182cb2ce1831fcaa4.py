"""Combined program: structured constructions + local search to minimize distinct cycle lengths."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_triangle(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    return True
    return False

def _min_degree_ok(adj, n, min_deg):
    return all(len(adj[i]) >= min_deg for i in range(n))

def _girth_ok(adj, n, min_girth):
    """Check if all cycles have length >= min_girth using BFS from each vertex."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node] and parent[neighbor] != node:
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _has_cycle(adj, n):
    """Check if graph has at least one cycle."""
    visited = [False] * n
    for start in range(n):
        if not visited[start]:
            stack = [(start, -1)]
            while stack:
                node, par = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for nb in adj[node]:
                    if nb == par:
                        continue
                    if visited[nb]:
                        return True
                    stack.append((nb, node))
    return False

def _count_distinct_cycle_lengths(adj, n):
    """Count distinct simple cycle lengths using BFS-based approach for efficiency."""
    cycle_lengths = set()
    # Use BFS from each vertex to find all cycles through that vertex
    for start in range(n):
        # BFS to find shortest path distances
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        
        # For each vertex, check if there's a cycle back to start
        # This is expensive, so use a different approach
        pass
    
    # Use DFS with early termination for small cycle lengths
    for start in range(n):
        visited = [False] * n
        visited[start] = True
        path = [start]
        found = set()
        
        def dfs(current, depth):
            if depth > 12:  # Max cycle length in 12-vertex graph
                return
            # Try to close cycle
            if depth >= 3 and start in adj[current]:
                found.add(depth)
            for next_node in adj[current]:
                if next_node == start:
                    continue
                if next_node > start and not visited[next_node]:
                    visited[next_node] = True
                    path.append(next_node)
                    dfs(next_node, depth + 1)
                    path.pop()
                    visited[next_node] = False
        
        dfs(start, 1)
        cycle_lengths.update(found)
    
    return len(cycle_lengths) if cycle_lengths else 0

def _evaluate(edges, n=12, min_deg=3, min_girth=4):
    """Returns (valid, num_distinct_cycle_lengths)."""
    edge_set = set()
    for u, v in edges:
        if u == v or u >= n or v >= n:
            return (False, 0)
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return (False, 0)
        edge_set.add(key)
    
    adj = _build_adj([(u, v) for u, v in edge_set], n)
    
    if not _min_degree_ok(adj, n, min_deg):
        return (False, 0)
    if _has_triangle(adj, n):
        return (False, 0)
    if not _girth_ok(adj, n, min_girth):
        return (False, 0)
    if not _has_cycle(adj, n):
        return (False, 0)
    
    ncl = _count_distinct_cycle_lengths(adj, n)
    return (True, ncl)

def _make_edges(edge_set):
    return [list(e) for e in sorted(edge_set)]

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill
    
    n = 12
    min_deg = 3
    min_girth = 4
    
    best_score = float('inf')
    best_edges = None
    
    # Strategy 1: Try known structured constructions
    constructions = []
    
    # Prism graph (current seed)
    es = set()
    for i in range(6):
        for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
            es.add(tuple(sorted((u, v))))
    constructions.append(es)
    
    # Two disjoint K4's won't work (degree 3 but no cross edges, disconnected is allowed)
    # Actually disconnected is allowed! Let's try two K4's
    # K4 has triangles, so invalid.
    
    # Try: two copies of a 4-vertex triangle-free min-degree-3 graph? 
    # Smallest triangle-free min-degree-3 graph is C4 (degree 2) - no.
    # K_{3,3} is 6 vertices, degree 3, girth 4. Two copies of K_{3,3}?
    # But we have 12 vertices total. Two K_{3,3} = 12 vertices!
    es2 = set()
    # First K_{3,3}: vertices 0,1,2 and 3,4,5
    for i in range(3):
        for j in range(3, 6):
            es2.add((i, j))
    # Second K_{3,3}: vertices 6,7,8 and 9,10,11
    for i in range(6, 9):
        for j in range(9, 12):
            es2.add((i, j))
    constructions.append(es2)
    
    # Try: Petersen-like. Petersen is 10 vertices. Add 2 more?
    # Let's try a 12-vertex graph based on two pentagons with cross connections
    # Or: C6 x K2 is the prism (already have it)
    
    # Try: 3-prism (two triangles with cross) - has triangles, invalid
    
    # Try: complete bipartite K_{4,4} + 4 isolated? No, min degree fails for isolated.
    # K_{4,4} on 8 vertices, plus 4 vertices connected to make min degree 3?
    
    # Try: C6 with each vertex connected to a "twin" (prism) - already have
    # Try: C8 + 4 vertices? 
    
    # Let me try a different approach: take a base graph and modify
    # Start with K_{3,3,3,3}? No, has triangles.
    
    # Try: 12-vertex graph from two hexagons with different cross-edges
    # Hexagon 1: 0-1-2-3-4-5-0, Hexagon 2: 6-7-8-9-10-11-6
    # Cross edges: 0-6, 1-7, 2-8, 3-9, 4-10, 5-11 (prism, already have)
    # Try twisted: 0-6, 1-8, 2-10, 3-11, 4-7, 5-9
    es3 = set()
    for i in range(6):
        es3.add((i, (i+1)%6))
        es3.add((i+6, (i+1)%6+6))
    # Twisted cross
    cross = [(0,6),(1,8),(2,10),(3,11),(4,7),(5,9)]
    for u,v in cross:
        es3.add((u,v))
    constructions.append(es3)
    
    # Evaluate all constructions
    for es in constructions:
        edges = _make_edges(es)
        valid, ncl = _evaluate(edges)
        if valid and ncl < best_score:
            best_score = ncl
            best_edges = edges[:]
    
    # Strategy 2: Random search with local improvement
    random.seed(12345)
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def _fix_min_degree(edge_set, n, min_deg):
        """Try to fix min degree by adding edges."""
        adj = _build_adj([(u,v) for u,v in edge_set], n)
        for _ in range(100):  # max iterations
            degrees = [len(adj[i]) for i in range(n)]
            if all(d >= min_deg for d in degrees):
                break
            # Find vertex with lowest degree
            min_v = min(range(n), key=lambda i: degrees[i])
            if degrees[min_v] >= min_deg:
                break
            # Try to add an edge from min_v to a random other vertex
            candidates = [u for u in range(n) if u != min_v and (min(min_v,u), max(min_v,u)) not in edge_set]
            if not candidates:
                break
            u = random.choice(candidates)
            edge_set.add((min(min_v,u), max(min_v,u)))
            adj = _build_adj([(a,b) for a,b in edge_set], n)
        return edge_set
    
    iterations = 0
    # If we find a graph with 1 distinct cycle length, we can stop immediately
    # as it is the theoretical minimum.
    while time.time() - start_time < DEADLINE and iterations < 50000:
        iterations += 1
        
        # Generate random graph with bias towards bipartite structure
        num_edges = random.randint(18, 28)
        edges_sample = random.sample(all_possible_edges, min(num_edges, len(all_possible_edges)))
        edge_set = set(edges_sample)
        
        # Fix min degree
        edge_set = _fix_min_degree(edge_set, n, min_deg)
        
        # Quick check
        adj = _build_adj([(u,v) for u,v in edge_set], n)
        if not _min_degree_ok(adj, n, min_deg):
            continue
        if _has_triangle(adj, n):
            continue
        if not _girth_ok(adj, n, min_girth):
            continue
        if not _has_cycle(adj, n):
            continue
        
        ncl = _count_distinct_cycle_lengths(adj, n)
        if ncl < best_score:
            best_score = ncl
            best_edges = _make_edges(edge_set)
            if best_score <= 2:
                break
        
        # Local improvement: try flipping edges
        if best_score <= 3:
            for _ in range(30):
                if time.time() - start_time > DEADLINE:
                    break
                # Try removing an edge
                if len(edge_set) > 18:
                    e = random.choice(list(edge_set))
                    edge_set.remove(e)
                    adj = _build_adj([(u,v) for u,v in edge_set], n)
                    if _min_degree_ok(adj, n, min_deg) and not _has_triangle(adj, n) and _girth_ok(adj, n, min_girth) and _has_cycle(adj, n):
                        ncl = _count_distinct_cycle_lengths(adj, n)
                        if ncl < best_score:
                            best_score = ncl
                            best_edges = _make_edges(edge_set)
                            if best_score == 1:
                                break
                        else:
                            edge_set.add(e)
                    else:
                        edge_set.add(e)
                # Try adding an edge
                if len(edge_set) < 30:
                    possible = [e for e in all_possible_edges if e not in edge_set]
                    if possible:
                        e = random.choice(possible)
                        edge_set.add(e)
                        adj = _build_adj([(u,v) for u,v in edge_set], n)
                        if _min_degree_ok(adj, n, min_deg) and not _has_triangle(adj, n) and _girth_ok(adj, n, min_girth) and _has_cycle(adj, n):
                            ncl = _count_distinct_cycle_lengths(adj, n)
                            if ncl < best_score:
                                best_score = ncl
                                best_edges = _make_edges(edge_set)
                                if best_score == 1:
                                    break
                            else:
                                edge_set.remove(e)
                        else:
                            edge_set.remove(e)
    
    if best_edges is None:
        # Fallback: return two disjoint K3,3 graphs (Value 2)
        es2 = set()
        for i in range(3):
            for j in range(3, 6):
                es2.add((i, j))
        for i in range(6, 9):
            for j in range(9, 12):
                es2.add((i, j))
        best_edges = _make_edges(es2)
    
    # Ensure proper types
    result = []
    for e in best_edges:
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int):
            continue
        if isinstance(u, bool) or isinstance(v, bool):
            continue
        result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
