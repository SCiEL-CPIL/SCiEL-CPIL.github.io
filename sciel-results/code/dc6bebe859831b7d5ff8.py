import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = [(min(u,v), max(u,v)) for u,v in P['edges']]
    m = len(edges)
    
    # Build adjacency: for each vertex, list of (edge_index, other_vertex)
    adj = {i: [] for i in range(n)}
    for ei, (u, v) in enumerate(edges):
        adj[u].append((ei, v))
        adj[v].append((ei, u))
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # CSP: assign each edge to color 0, 1, or 2
    # Constraint: each vertex has exactly 2 edges of each color
    # Use backtracking with degree-based pruning
    
    edge_color = [-1] * m  # -1 = unassigned
    # For each vertex and color, count how many edges of that color are assigned
    vert_color_count = [[0, 0, 0] for _ in range(n)]
    
    def extract_cycle(color):
        """Extract the cycle from edges of given color. Returns list of vertices or None."""
        # Build adjacency for this color
        color_adj = {i: [] for i in range(n)}
        for ei, (u, v) in enumerate(edges):
            if edge_color[ei] == color:
                color_adj[u].append(v)
                color_adj[v].append(u)
        # Each vertex should have degree 2
        for i in range(n):
            if len(color_adj[i]) != 2:
                return None
        # Walk the cycle
        if not color_adj[0]:
            return None
        cycle = [0]
        visited = {0}
        curr = 0
        prev = -1
        for _ in range(n):
            nxts = color_adj[curr]
            if len(nxts) == 2:
                nxt = nxts[0] if nxts[0] != prev else nxts[1]
            else:
                return None
            if nxt == 0 and len(cycle) < n:
                return None
            if nxt in visited and nxt != 0:
                return None
            if nxt == 0:
                break
            cycle.append(nxt)
            visited.add(nxt)
            prev = curr
            curr = nxt
        if len(cycle) != n:
            return None
        return cycle
    
    def try_solve():
        """Backtracking solver for the 3-edge-coloring CSP."""
        # Find unassigned edge with fewest options (most constrained)
        def find_next_edge():
            best_ei = -1
            best_options = 4
            for ei in range(m):
                if edge_color[ei] == -1:
                    u, v = edges[ei]
                    options = 0
                    for c in range(3):
                        if vert_color_count[u][c] < 2 and vert_color_count[v][c] < 2:
                            options += 1
                    if options < best_options:
                        best_options = options
                        best_ei = ei
                        if options == 0:
                            return best_ei
                    if options <= 1:
                        break
            return best_ei
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            ei = find_next_edge()
            if ei == -1:
                # All edges assigned, verify
                for c in range(3):
                    cyc = extract_cycle(c)
                    if cyc is None:
                        return False
                return True
            
            u, v = edges[ei]
            # Try colors in order of least constraining value
            colors = []
            for c in range(3):
                if vert_color_count[u][c] < 2 and vert_color_count[v][c] < 2:
                    # Score: prefer colors that leave more flexibility
                    score = (2 - vert_color_count[u][c]) + (2 - vert_color_count[v][c])
                    colors.append((score, c))
            colors.sort(reverse=True)
            
            for _, c in colors:
                edge_color[ei] = c
                vert_color_count[u][c] += 1
                vert_color_count[v][c] += 1
                
                # Check if any vertex is now stuck (can't complete)
                valid = True
                for vertex in [u, v]:
                    for cc in range(3):
                        remaining = 2 - vert_color_count[vertex][cc]
                        if remaining < 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    if backtrack():
                        return True
                
                vert_color_count[u][c] -= 1
                vert_color_count[v][c] -= 1
                edge_color[ei] = -1
            
            return False
        
        return backtrack()
    
    if try_solve():
        result = []
        for c in range(3):
            cyc = extract_cycle(c)
            if cyc is not None:
                result.append([int(v) for v in cyc])
        if len(result) == 3:
            return result
    
    # Fallback: sequential Hamiltonian cycle search
    remaining_edges = set(edges)
    adj2 = {i: set() for i in range(n)}
    for u, v in edges:
        adj2[u].add(v)
        adj2[v].add(u)
    
    cycles_found = []
    
    def find_hamiltonian_cycle():
        path = [0]
        visited = {0}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                edge_key = (0, last) if 0 < last else (last, 0)
                if edge_key in remaining_edges:
                    return path[:]
                return None
            
            curr = path[-1]
            neighbors = list(adj2[curr])
            neighbors.sort(key=lambda x: len(adj2[x]))
            
            for nxt in neighbors:
                if nxt not in visited:
                    edge_key = (curr, nxt) if curr < nxt else (nxt, curr)
                    if edge_key in remaining_edges:
                        path.append(nxt)
                        visited.add(nxt)
                        
                        dead_end = False
                        unvisited = [v for v in range(n) if v not in visited]
                        if len(unvisited) > 1:
                            for v in unvisited:
                                has_unvisited_neighbor = False
                                for u in adj2[v]:
                                    if u in unvisited:
                                        has_unvisited_neighbor = True
                                        break
                                if not has_unvisited_neighbor:
                                    dead_end = True
                                    break
                        
                        if not dead_end:
                            res = backtrack()
                            if res is not None:
                                return res
                        
                        path.pop()
                        visited.remove(nxt)
            
            return None
        
        return backtrack()
    
    while True:
        if time.time() > deadline:
            break
        
        cycle = find_hamiltonian_cycle()
        if cycle is None:
            break
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = (u, v) if u < v else (v, u)
            if edge_key in remaining_edges:
                remaining_edges.remove(edge_key)
                adj2[u].remove(v)
                adj2[v].remove(u)
        
        cycles_found.append(cycle)
        if len(cycles_found) >= 3:
            break
    
    if not cycles_found:
        return [list(range(n))]
    
    return [[int(v) for v in c] for c in cycles_found]

# EVOLVE-BLOCK-END
