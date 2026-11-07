"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 80  # safety margin before 90s kill

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_index(e):
    """Convert frozenset edge to index."""
    u, v = tuple(sorted(e))
    return (u, v)

def _find_3_hamiltonian_cycles(adj, deadline):
    """
    Try to find 3 edge-disjoint Hamiltonian cycles using 3-edge-coloring approach.
    Each color class must form a single cycle visiting all 21 vertices.
    """
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u,v), max(u,v))
        edge_to_idx[key] = i
    
    # For each vertex, list of incident edges
    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    # color[i] = color of edge i, -1 if uncolored
    color = [-1] * m
    
    # For each color, track which vertices have degree 0, 1, 2 in that color
    # degree[color][v] = number of edges of that color incident to v
    degree = [[0] * n for _ in range(3)]
    
    # For connectivity check: for each color, we need to verify it's a single cycle
    # We'll check at the end
    
    def is_valid_coloring():
        """Check if current partial coloring is still valid."""
        for c in range(3):
            for v in range(n):
                if degree[c][v] > 2:
                    return False
        return True
    
    def check_complete():
        """Check if coloring is complete and each color forms a Hamiltonian cycle."""
        for c in range(3):
            # Check all vertices have degree 2
            for v in range(n):
                if degree[c][v] != 2:
                    return False
            # Check it's a single cycle (not multiple cycles)
            # Build adjacency for this color
            adj_c = [set() for _ in range(n)]
            for i, (u, v) in enumerate(edges):
                if color[i] == c:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
            # BFS/DFS to check connectivity
            visited = [False] * n
            visited[0] = True
            stack = [0]
            count = 1
            while stack:
                node = stack.pop()
                for nb in adj_c[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        count += 1
                        stack.append(nb)
            if count != n:
                return False
        return True
    
    def get_cycle_for_color(c):
        """Reconstruct cycle for color c."""
        adj_c = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            if color[i] == c:
                adj_c[u].add(v)
                adj_c[v].add(u)
        # Find cycle starting from 0
        cycle = [0]
        visited = {0}
        current = 0
        for _ in range(n - 1):
            nexts = [nb for nb in adj_c[current] if nb not in visited]
            if not nexts:
                # Should not happen if valid
                break
            current = nexts[0]
            cycle.append(current)
            visited.add(current)
        return cycle
    
    # Backtracking with symmetry breaking
    # Fix edge 0 to color 0
    def backtrack(edge_idx):
        if time.time() > deadline:
            return False
        
        if edge_idx == m:
            return check_complete()
        
        if not is_valid_coloring():
            return False
        
        u, v = edges[edge_idx]
        
        for c in range(3):
            # Check if assigning color c is valid
            if degree[c][u] >= 2 or degree[c][v] >= 2:
                continue
            
            # Symmetry breaking: for the first few edges, restrict colors
            # to reduce search space
            if edge_idx == 0:
                if c != 0:
                    continue
            elif edge_idx == 1:
                if c < 1:
                    continue
            
            # Assign
            color[edge_idx] = c
            degree[c][u] += 1
            degree[c][v] += 1
            
            if backtrack(edge_idx + 1):
                return True
            
            # Unassign
            color[edge_idx] = -1
            degree[c][u] -= 1
            degree[c][v] -= 1
        
        return False
    
    if backtrack(0):
        cycles = []
        for c in range(3):
            cycle = get_cycle_for_color(c)
            cycles.append(cycle)
        return cycles
    
    return None

def _find_cycles_greedy(adj, S, max_cycles, deadline):
    """
    Find edge-disjoint cycles all using exactly vertex set S.
    Returns list of cycles (each a list of vertices).
    """
    S_set = frozenset(S)
    S_list = list(S)
    k = len(S_list)
    
    # Precompute: for each vertex in S, neighbors in S
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
    
    used_edges = set()
    cycles = []
    
    # Try to find cycles greedily with backtracking
    def dfs_cycle(path, visited, local_used_edges):
        if time.time() > deadline:
            return None
        
        if len(path) == k:
            # Check if we can close the cycle
            last = path[-1]
            if last in adj_S[S_list[0]] and frozenset([last, S_list[0]]) not in local_used_edges and frozenset([last, S_list[0]]) not in used_edges:
                # Valid cycle
                new_used = local_used_edges | {frozenset([last, S_list[0]])}
                return (path, new_used)
            return None
        
        for neighbor in sorted(adj_S[path[-1]]):
            if neighbor not in visited:
                edge = frozenset([path[-1], neighbor])
                if edge not in used_edges and edge not in local_used_edges:
                    visited.add(neighbor)
                    local_used_edges.add(edge)
                    result = dfs_cycle(path + [neighbor], visited, local_used_edges)
                    if result is not None:
                        return result
                    local_used_edges.discard(edge)
                    visited.discard(neighbor)
        return None
    
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        start = S_list[0]
        visited = {start}
        local_used = set()
        result = dfs_cycle([start], visited, local_used)
        
        if result is not None:
            cycle, new_used = result
            cycles.append(cycle)
            used_edges = used_edges | new_used
        else:
            break
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    adj = _build_adjacency()
    n = P['n']
    
    best_cycles = None
    best_count = 0
    
    # First try the 3-edge-coloring approach for Hamiltonian cycles on all 21 vertices
    cycles_3 = _find_3_hamiltonian_cycles(adj, deadline)
    if cycles_3 is not None and len(cycles_3) == 3:
        best_cycles = [list(c) for c in cycles_3]
        best_count = 3
    
    # If we didn't find 3, try other approaches
    if best_count < 3:
        # Try to find cycles on all 21 vertices using greedy approach
        all_vertices = list(range(n))
        cycles_greedy = _find_cycles_greedy(adj, all_vertices, 3, deadline)
        if cycles_greedy and len(cycles_greedy) > best_count:
            best_cycles = [list(c) for c in cycles_greedy]
            best_count = len(cycles_greedy)
    
    # Try smaller vertex sets
    for s in [7, 6, 5, 4, 3]:
        if time.time() > deadline - 5:
            break
        
        max_possible = min(10, 63 // s)
        if max_possible <= best_count:
            continue
        
        all_vertices = list(range(n))
        candidates = list(itertools.combinations(all_vertices, s))[:500]
        
        for candidate in candidates:
            if time.time() > deadline - 3:
                break
            
            S = list(candidate)
            cycles = _find_cycles_greedy(adj, S, max_possible, deadline)
            
            if cycles and len(cycles) > best_count:
                best_cycles = [list(c) for c in cycles]
                best_count = len(cycles)
                
                if best_count >= max_possible:
                    break
        
        if best_count >= 10:
            break
    
    # Ensure we have at least one valid cycle
    if best_cycles is None or len(best_cycles) == 0:
        # Fallback: find any single triangle
        edge_set = set()
        for u, v in P['edges']:
            edge_set.add(frozenset([u, v]))
        
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    if frozenset([w, u]) in edge_set and frozenset([w, v]) in edge_set:
                        best_cycles = [[u, v, w]]
                        best_count = 1
                        break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        return [list(range(P['n']))]
    
    # Validate and convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
