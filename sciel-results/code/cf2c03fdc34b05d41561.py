import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 145.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _validate(paths, edges):
        """Check that paths are simple, use graph edges, are edge-disjoint and cover all edges."""
        if not paths:
            return len(edges) == 0
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e in used:
                    return False
                used.add(e)
        return used == set(edges)

    def _greedy_decompose(edges, n, max_time=3.0):
        """Greedy path decomposition with random restarts."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 2.0:
            remaining = set(edge_set)
            paths = []
            
            while remaining:
                # Compute remaining degrees
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                
                # Start from an odd-degree vertex if possible
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                
                if rem_odd:
                    start_v = random.choice(rem_odd)
                else:
                    candidates = [v for v in rem_deg if rem_deg[v] > 0]
                    if not candidates:
                        break
                    start_v = random.choice(candidates)
                
                path = _extend_path_heuristic(start_v, remaining, adj, n)
                
                if not path or len(path) < 2:
                    # Fallback: pick any edge from start_v
                    found = False
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            found = True
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            found = True
                            break
                    if not found:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
        
        if best_paths is None:
            return [[u, v] for u, v in sorted(edge_set)]
        return best_paths

    def _extend_path_heuristic(start, remaining, adj, n):
        """Extend a path using minimum remaining degree heuristic."""
        path = [start]
        visited = {start}
        current = start
        
        while True:
            available = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if e in remaining and nb not in visited:
                    # Count available edges from nb (excluding current)
                    deg = 0
                    for x in adj[nb]:
                        if x == current:
                            continue
                        e2 = tuple(sorted((nb, x)))
                        if e2 in remaining and x not in visited:
                            deg += 1
                    available.append((deg, nb))
            
            if not available:
                break
            
            # Pick neighbor with minimum remaining degree
            available.sort()
            min_deg = available[0][0]
            candidates = [nb for deg, nb in available if deg == min_deg]
            best_nb = random.choice(candidates)
            
            e = tuple(sorted((current, best_nb)))
            remaining.discard(e)
            path.append(best_nb)
            visited.add(best_nb)
            current = best_nb
        
        return path

    def _find_2factor_paths(edges, n, max_time=5.0):
        """
        Find a 2-factor (collection of cycles and paths covering all vertices with deg <= 2)
        using greedy matching on the line graph, then convert to paths.
        
        This creates long paths/cycles that we then break.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        m = len(edge_set)
        if m == 0:
            return []
        
        edges_list = sorted(edge_set)
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build line graph adjacency
        edge_to_idx = {e: i for i, e in enumerate(edges_list)}
        adj_line = [set() for _ in range(m)]
        
        # For each vertex, all edges incident to it form a clique in the line graph
        for v in range(n):
            incident = [e for e in edge_set if v in e]
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    ei = edge_to_idx[incident[i]]
                    ej = edge_to_idx[incident[j]]
                    adj_line[ei].add(ej)
                    adj_line[ej].add(ei)
        
        # Greedy matching on line graph to create long chains
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 3.0:
            match = [-1] * m
            matched = [False] * m
            
            # Process edges in random order, preferring high-degree vertices
            order = list(range(m))
            random.shuffle(order)
            
            # Sort by vertex degree of the edge's endpoints (prefer edges at high-degree vertices)
            def edge_priority(ei):
                u, v = edges_list[ei]
                return -(len(adj[u]) + len(adj[v]))
            
            order.sort(key=lambda ei: edge_priority(ei))
            random.shuffle(order)  # Add randomness to tie-breaking
            
            for i in order:
                if matched[i]:
                    continue
                candidates = [j for j in adj_line[i] if not matched[j]]
                if candidates:
                    # Pick the most constrained candidate
                    best_j = min(candidates, key=lambda j: len([k for k in adj_line[j] if not matched[k]]))
                    match[i] = best_j
                    match[best_j] = i
                    matched[i] = True
                    matched[best_j] = True
            
            # Reconstruct paths from matching
            paths = _reconstruct_paths_from_matching(edges_list, match, n)
            
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
        
        if best_paths is None:
            return _greedy_decompose(edges, n, max_time=max_time)
        return best_paths

    def _reconstruct_paths_from_matching(edges_list, match, n):
        """
        Given a matching on the line graph, reconstruct the paths.
        Each matched pair (e1, e2) means these edges are consecutive in a path.
        Unmatched edges are path endpoints.
        """
        m = len(edges_list)
        
        # Build adjacency: for each edge, which edge it's connected to
        # And determine direction
        # edge_i connects to edge_j if match[i] == j
        
        # Each edge has two vertices. If edge_i = (u,v) and edge_j = (v,w), 
        # then they connect at vertex v.
        
        # Build a graph where edges are nodes, and matching edges connect them
        # We need to trace paths through this graph
        
        # For each edge, determine its "position" in the path
        # An edge (u,v) can connect to another edge at u or at v
        
        # Create a directed structure
        # For each edge i, if match[i] != -1, then i is connected to match[i]
        # We need to figure out the vertex where they connect
        
        # Build: for each edge, store which vertex it uses to connect to its match
        # edge_connect[i] = vertex where edge i connects to its matched edge
        
        edge_connect = [None] * m  # vertex where this edge connects to its match
        edge_start = [None] * m    # start vertex of the path segment
        edge_end = [None] * m      # end vertex of the path segment
        
        for i in range(m):
            if match[i] == -1:
                # This is an endpoint
                continue
            j = match[i]
            u1, v1 = edges_list[i]
            u2, v2 = edges_list[j]
            
            # Find common vertex
            if u1 == u2:
                common = u1
                edge_connect[i] = common
                edge_connect[j] = common
            elif u1 == v2:
                common = u1
                edge_connect[i] = common
                edge_connect[j] = common
            elif v1 == u2:
                common = v1
                edge_connect[i] = common
                edge_connect[j] = common
            elif v1 == v2:
                common = v1
                edge_connect[i] = common
                edge_connect[j] = common
            else:
                # Should not happen if matching is valid
                continue
        
        # Now trace paths
        # A path starts at an unmatched edge or an edge that is the "start" of a chain
        # We need to find the start of each chain
        
        # Build next pointers: from edge i, where do we go?
        # If edge i is matched to edge j, and they connect at vertex c,
        # then from i we go to j, and the direction is determined by which vertex of j is not c
        
        next_edge = [None] * m
        next_vertex = [None] * m  # the vertex we arrive at in the next edge
        
        for i in range(m):
            if match[i] == -1:
                continue
            j = match[i]
            c = edge_connect[i]
            if c is None:
                continue
            
            u, v = edges_list[j]
            if u == c:
                next_edge[i] = j
                next_vertex[i] = v
            elif v == c:
                next_edge[i] = j
                next_vertex[i] = u
        
        # Find all path starts: edges that are not "continued" from another edge
        # An edge i is a start if there's no edge k such that next_edge[k] == i
        is_continued = [False] * m
        for i in range(m):
            if next_edge[i] is not None:
                is_continued[next_edge[i]] = True
        
        paths = []
        visited_edges = set()
        
        for i in range(m):
            if not is_continued[i] and i not in visited_edges:
                # Start a new path from edge i
                path_vertices = []
                cur_edge = i
                cur_start_vertex = None
                cur_end_vertex = None
                
                u, v = edges_list[i]
                
                # Determine direction: if this edge is a start, we need to figure out
                # which vertex is the "start" and which is the "end"
                # If the edge is matched, the "end" is where it connects to the next edge
                if match[i] != -1 and edge_connect[i] is not None:
                    c = edge_connect[i]
                    if u == c:
                        cur_start_vertex = v
                        cur_end_vertex = u
                    else:
                        cur_start_vertex = u
                        cur_end_vertex = v
                else:
                    # Unmatched edge, just use (u, v)
                    cur_start_vertex = u
                    cur_end_vertex = v
                
                path_vertices = [cur_start_vertex, cur_end_vertex]
                visited_edges.add(i)
                
                while next_edge[cur_edge] is not None:
                    next_e = next_edge[cur_edge]
                    if next_e in visited_edges:
                        break
                    visited_edges.add(next_e)
                    
                    nv = next_vertex[cur_edge]
                    # The next edge starts at nv
                    u2, v2 = edges_list[next_e]
                    if u2 == nv:
                        path_vertices.append(v2)
                    elif v2 == nv:
                        path_vertices.append(u2)
                    
                    cur_edge = next_e
                
                if len(path_vertices) >= 2 and len(set(path_vertices)) == len(path_vertices):
                    paths.append(path_vertices)
        
        return paths

    def _solve_one(edges, n, max_time=10.0):
        """Solve one instance using multiple strategies."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Strategy 1: Greedy with random restarts
        greedy_paths = _greedy_decompose(edges, n, max_time=min(3.0, max_time / 3))
        
        # Strategy 2: 2-factor based approach
        factor_paths = _find_2factor_paths(edges, n, max_time=min(4.0, max_time / 2))
        
        # Pick the best
        best = greedy_paths
        if factor_paths and len(factor_paths) < len(greedy_paths):
            if _validate(factor_paths, edges):
                best = factor_paths
        
        # Validate and return
        if _validate(best, edges):
            return best
        
        # Fallback
        return [[u, v] for u, v in sorted(edge_set)]

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        
        # Allocate time based on remaining budget
        remaining = _remaining()
        if remaining < 5.0:
            max_t = max(1.0, remaining - 1.0)
        else:
            max_t = min(40.0, remaining / (len(specs) - i))
        
        paths = _solve_one(edges, n, max_time=max_t)
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
