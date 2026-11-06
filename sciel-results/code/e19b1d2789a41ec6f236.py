import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_matching_bipartite(adj):
    """
    Finds maximum matching in a bipartite graph.
    adj[u] = list of v such that (u, v) is an edge.
    Returns a list match where match[u] = v if u is matched to v, else -1.
    u is in left partition, v is in right partition.
    """
    n_left = len(adj)
    match = [-1] * n_left
    match_r = [-1] * n_left  # Assuming square or at least same size for simplicity, here we have M left and M right
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match[u] = v
                match_r[v] = u
                return True
        return False

    result = 0
    for u in range(n_left):
        visited = [False] * n_left
        if dfs(u, visited):
            result += 1
            
    return match, match_r

def _decompose(edges, n, seed, time_limit=100.0):
    if not edges:
        return []
    
    start_time = time.time()
    
    # 1. Build Bipartite Graph
    # Left nodes: 0..M-1, Right nodes: 0..M-1
    M = len(edges)
    adj = [[] for _ in range(M)]
    
    # Map edge index to its endpoints
    edge_endpoints = edges
    
    # For each vertex, find all incident edges
    vertex_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_to_edges[u].append(i)
        vertex_to_edges[v].append(i)
        
    # Add edges to bipartite graph
    # If edge i and edge j share a vertex, we can match i (left) to j (right)
    # We must avoid i == j
    for v in vertex_to_edges:
        e_list = vertex_to_edges[v]
        for i in e_list:
            for j in e_list:
                if i != j:
                    # i in Left, j in Right
                    adj[i].append(j)
                    
    # 2. Find Maximum Matching
    match_l, match_r = _max_matching_bipartite(adj)
    
    # 3. Reconstruct Pieces
    # match_l[i] = j means edge i is followed by edge j
    # match_r[j] = i means edge j is preceded by edge i
    
    visited_edges = [False] * M
    pieces = []
    
    for i in range(M):
        if visited_edges[i]:
            continue
            
        # Start a new path/cycle from i
        # Check if i has a predecessor
        if match_r[i] != -1:
            # i is not a start, skip (it will be reached from its predecessor)
            continue
            
        # Start at i
        path_edges = [i]
        visited_edges[i] = True
        current = i
        
        while True:
            next_edge = match_l[current]
            if next_edge == -1:
                # End of path
                break
            if visited_edges[next_edge]:
                # Cycle detected
                # The cycle is path_edges + [next_edge] ... wait
                # If next_edge is already in path_edges, we have a cycle.
                # But in a matching, each node has at most one successor and one predecessor.
                # So if we hit a visited node, it must be the start of the current path?
                # Or it's a cycle that doesn't include the start?
                # Since we started from a node with no predecessor, we can only form a cycle
                # if we come back to the start.
                if next_edge == i:
                    # It's a cycle
                    break
                else:
                    # This shouldn't happen in a valid matching if we started correctly
                    # unless there's a bug in matching logic
                    break
            
            path_edges.append(next_edge)
            visited_edges[next_edge] = True
            current = next_edge
            
        # Convert path_edges to vertices
        if len(path_edges) == 1:
            u, v = edge_endpoints[path_edges[0]]
            pieces.append([u, v])
        else:
            vertices = []
            # Determine orientation
            # First edge e0
            e0 = edge_endpoints[path_edges[0]]
            # If path has more than 1 edge, e0 must be oriented towards the shared vertex with e1
            if len(path_edges) > 1:
                e1 = edge_endpoints[path_edges[1]]
                # Find shared vertex
                shared = None
                for u in e0:
                    if u in e1:
                        shared = u
                        break
                # e0 ends at shared. So start is the other end of e0.
                start_v = e0[0] if e0[1] == shared else e0[1]
                vertices.append(start_v)
                vertices.append(shared)
                
                # Subsequent edges
                for k in range(1, len(path_edges)):
                    ek = edge_endpoints[path_edges[k]]
                    prev_shared = vertices[-1]
                    # ek starts at prev_shared
                    # Find the other end
                    end_v = ek[0] if ek[1] == prev_shared else ek[1]
                    vertices.append(end_v)
                    
                    # If this is the last edge, and it's a cycle, it should connect back to start_v
                    # But we just append the end. The "closing" is implied by the piece type.
                    # However, we need to check if it's a cycle to report it as such?
                    # The task says: "A piece is a single edge [u, v] or a cycle given by at least three distinct vertices".
                    # It doesn't explicitly say "path".
                    # "Pieces must be edge-disjoint and cover all edges."
                    # "Minimize the total number of pieces."
                    # If the matching produces a path (not a cycle), is it a valid piece?
                    # "A piece is a single edge [u, v] or a cycle..."
                    # This implies PATHS ARE NOT VALID PIECES unless they are single edges or cycles.
                    # WAIT. The task description says: "A piece is a single edge [u, v] or a cycle given by at least three distinct vertices".
                    # This means we CANNOT have a path of length > 1 as a single piece!
                    # We must decompose into CYCLES and SINGLE EDGES.
                    # My matching approach minimizes the number of PATHS and CYCLES.
                    # If the result contains paths of length > 1, they are INVALID.
                    
            # Let's re-read carefully.
            # "A piece is a single edge [u, v] or a cycle given by at least three distinct vertices (the closing edge is implied)."
            # This restricts the pieces to Cycles and Edges.
            # So if the matching yields a path $e_1-e_2$, it is NOT a valid piece.
            # We would have to split it into two single edges? Or find a cycle?
            # The matching minimizes the number of components in the "path/cycle cover".
            # If a component is a path, it contributes 1 to the count.
            # But if paths are not allowed, we must convert them to edges?
            # A path of length $k$ has $k$ edges. If we split it into $k$ single edges, the count increases.
            # So the matching approach gives a lower bound, but might not be directly usable if paths are forbidden.
            
            # However, in many such problems, "path" is allowed.
            # Let's check the seed program.
            # Seed program: `pieces.append(cycle)` or `pieces.append([u, v])`.
            # It only appends cycles or single edges.
            # So paths of length > 1 are NOT allowed.
            
            # This changes the problem significantly.
            # We need to decompose into Cycles and Edges.
            # This is the "Cycle and Edge Decomposition" problem.
            # It is known that any graph can be decomposed into edge-disjoint cycles and paths.
            # But if paths are not allowed, we must decompose into cycles and single edges.
            # This is possible if and only if... ?
            # Actually, any edge not in a cycle must be a single edge.
            # So the optimal strategy is:
            # 1. Find a set of edge-disjoint cycles.
            # 2. All remaining edges are single edges.
            # 3. Maximize the number of edges covered by cycles?
            # No, minimize total pieces.
            # A cycle of length $k$ covers $k$ edges with 1 piece.
            # A single edge covers 1 edge with 1 piece.
            # So we want to maximize the number of edges that are part of cycles?
            # Not exactly. A cycle of length 3 saves 2 pieces compared to 3 edges.
            # A cycle of length 4 saves 3 pieces.
            # So we want to find cycles that are as long as possible?
            # Or just find a maximum set of edge-disjoint cycles?
            # If we find a set of edge-disjoint cycles, the remaining edges are single edges.
            # Total pieces = (# cycles) + (# remaining edges).
            # Let $E$ be total edges.
            # Let $C$ be the set of cycles.
            # Let $E_C$ be the number of edges in $C$.
            # Remaining edges = $E - E_C$.
            # Total pieces = $|C| + (E - E_C)$.
            # To minimize this, we want to maximize $E_C - |C|$.
            # Since $E_C = \sum_{c \in C} \text{len}(c)$,
            # We want to maximize $\sum (\text{len}(c) - 1)$.
            # This is equivalent to finding a set of edge-disjoint cycles that maximizes the sum of (length - 1).
            # Since length >= 3, this is roughly maximizing the total length of cycles.
            
            # So the problem reduces to: Find a set of edge-disjoint cycles maximizing the sum of their lengths.
            # This is a weighted maximum set packing problem on cycles, which is NP-hard in general.
            # However, for small graphs, we can use heuristics.
            
            # Given the time limit, a greedy approach of finding the longest available cycle repeatedly is a good heuristic.
            
            # Let's revert to the greedy cycle peeling approach but optimize it.
            pass

    # Re-implementing the greedy approach with better cycle finding
    def _find_longest_cycle(adj, seed):
        if not adj:
            return []
        
        vertices = [v for v in adj if adj[v]]
        if not vertices:
            return []
        
        rng = random.Random(seed)
        best_cycle = []
        
        # Try starting from high degree vertices
        deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
        starts = deg_sorted[:10]
        rng.shuffle(starts)
        
        for start in starts:
            if not adj[start]:
                continue
            
            visited = {start}
            path = [start]
            current = start
            
            # DFS with backtracking for small depth?
            # Or simple greedy?
            # Greedy: always go to unvisited neighbor with highest degree
            while True:
                unvisited = [v for v in adj[current] if v not in visited]
                if not unvisited:
                    break
                
                # Check if we can close
                if start in adj[current] and len(path) >= 3:
                    if len(path) > len(best_cycle):
                        best_cycle = list(path)
                
                unvisited.sort(key=lambda v: -len(adj[v]))
                next_v = unvisited[0]
                
                visited.add(next_v)
                path.append(next_v)
                current = next_v
                
        return best_cycle

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    remaining_edges = set(edges)
    pieces = []
    
    cycle_seed = 0
    while remaining_edges:
        # Rebuild adj for remaining edges?
        # Or maintain it dynamically.
        # Dynamic is better.
        
        # Find longest cycle in current adj
        cycle = _find_longest_cycle(adj, cycle_seed)
        cycle_seed += 1
        
        if cycle:
            # Extract edges
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = (min(u,v), max(u,v))
                cycle_edges.add(e)
                
            # Remove
            for e in cycle_edges:
                remaining_edges.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                
            pieces.append(cycle)
        else:
            # No cycle found, remove a single edge
            # Pick an edge with low degree endpoints to preserve cycles
            best_edge = None
            min_deg = float('inf')
            for u, v in remaining_edges:
                deg = len(adj[u]) + len(adj[v])
                if deg < min_deg:
                    min_deg = deg
                    best_edge = (u, v)
                    
            if best_edge:
                u, v = best_edge
                remaining_edges.discard(best_edge)
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
            else:
                break

    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed)
        results.append(pieces)
    return results
