"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import sys
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency structure
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    adj = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each edge, the edge index
    def get_edge_idx(u, v):
        return edge_map[(u, v)]
    
    def solve_csp():
        """
        Solve the edge-coloring CSP: assign each edge to one of 3 cycle groups
        such that every vertex has exactly 2 incident edges per group.
        Then verify each group forms a single Hamiltonian cycle.
        """
        # edge_color[e] = 0, 1, 2 or -1 for unassigned
        edge_color = [-1] * m
        
        # For each vertex and color, count assigned edges
        # v_color_count[v][c] = number of edges of color c incident to v
        v_color_count = [[0, 0, 0] for _ in range(n)]
        
        # For each vertex and color, the set of edges with that color incident to v
        v_color_edges = [set() for _ in range(n)]  # v_color_edges[v] will hold dict color->set
        v_color_edges = [defaultdict(set) for _ in range(n)]
        
        def can_assign(e, c):
            """Check if edge e can be assigned color c."""
            u, v = edges[e]
            # Vertex u needs: v_color_count[u][c] < 2
            if v_color_count[u][c] >= 2:
                return False
            # Vertex v needs: v_color_count[v][c] < 2
            if v_color_count[v][c] >= 2:
                return False
            return True
        
        def assign(e, c):
            """Assign edge e color c."""
            u, v = edges[e]
            edge_color[e] = c
            v_color_count[u][c] += 1
            v_color_count[v][c] += 1
            v_color_edges[u][c].add(e)
            v_color_edges[v][c].add(e)
        
        def unassign(e, c):
            """Unassign edge e from color c."""
            u, v = edges[e]
            edge_color[e] = -1
            v_color_count[u][c] -= 1
            v_color_count[v][c] -= 1
            v_color_edges[u][c].discard(e)
            v_color_edges[v][c].discard(e)
        
        def arc_consistency():
            """
            Apply arc consistency: if a vertex v has only one available color
            for its remaining degree-2 requirement, force assignments.
            Returns False if contradiction detected.
            """
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    for c in range(3):
                        # How many more edges of color c does v need?
                        need = 2 - v_color_count[v][c]
                        if need < 0:
                            return False
                        if need == 0:
                            # v already has 2 edges of color c; ensure no unassigned edge
                            # incident to v can be assigned color c
                            for e in range(m):
                                if edge_color[e] == -1 and v in edges[e]:
                                    if edges[e][0] == v or edges[e][1] == v:
                                        # This edge cannot be color c
                                        # We don't track domains explicitly, but we can
                                        # check during assignment
                                        pass
                        elif need == 1:
                            # v needs exactly 1 more edge of color c
                            # Count unassigned edges incident to v
                            unassigned_incident = []
                            for e in range(m):
                                if edge_color[e] == -1:
                                    u2, v2 = edges[e]
                                    if u2 == v or v2 == v:
                                        unassigned_incident.append(e)
                            if len(unassigned_incident) < need:
                                return False
                            # If exactly one unassigned edge can take color c (others blocked),
                            # we could force it, but let's keep it simple for now
                        else:  # need == 2
                            unassigned_incident = []
                            for e in range(m):
                                if edge_color[e] == -1:
                                    u2, v2 = edges[e]
                                    if u2 == v or v2 == v:
                                        unassigned_incident.append(e)
                            if len(unassigned_incident) < need:
                                return False
            return True
        
        def find_ham_cycle(color_c):
            """Find the cycle formed by edges of color c. Returns list of vertices or None."""
            # Build adjacency using only edges of color c
            adj_c = defaultdict(set)
            for e in range(m):
                if edge_color[e] == color_c:
                    u, v = edges[e]
                    adj_c[u].add(v)
                    adj_c[v].add(u)
            
            # Check all vertices have degree 2
            for v in range(n):
                if len(adj_c[v]) != 2:
                    return None
            
            # Traverse the cycle
            path = [0]
            visited = {0}
            curr = 0
            while True:
                nexts = list(adj_c[curr])
                next_node = None
                for nb in nexts:
                    if nb not in visited:
                        next_node = nb
                        break
                if next_node is None:
                    if len(path) == n and 0 in adj_c[curr]:
                        return path
                    else:
                        return None
                visited.add(next_node)
                path.append(next_node)
                curr = next_node
        
        def backtrack():
            """Backtracking search to assign colors to edges."""
            if time.time() > deadline:
                return None
            
            # Find the next unassigned edge
            e = -1
            min_unassigned = None
            for i in range(m):
                if edge_color[i] == -1:
                    # Prefer edges where one endpoint has limited options
                    u, v = edges[i]
                    # Count available colors for this edge
                    avail_u = 3 - v_color_count[u][0] - (1 if v_color_count[u][0] >= 2 else 0)
                    # Simpler: just pick the first unassigned edge, or use MRV
                    e = i
                    break
            
            if e == -1:
                # All edges assigned; verify Hamiltonian cycles
                c0 = find_ham_cycle(0)
                c1 = find_ham_cycle(1)
                c2 = find_ham_cycle(2)
                if c0 is not None and c1 is not None and c2 is not None:
                    return [c0, c1, c2]
                return None
            
            # Try assigning colors to edge e
            u, v = edges[e]
            # Order colors by how constrained they are
            colors_to_try = []
            for c in range(3):
                if can_assign(e, c):
                    # Prefer colors where the vertex is more constrained
                    constraint_score = v_color_count[u][c] + v_color_count[v][c]
                    colors_to_try.append((constraint_score, c))
            colors_to_try.sort()
            
            for _, c in colors_to_try:
                if time.time() > deadline:
                    return None
                assign(e, c)
                if arc_consistency():
                    result = backtrack()
                    if result is not None:
                        return result
                unassign(e, c)
            
            return None
        
        return backtrack()
    
    # Try the CSP approach
    result = solve_csp()
    
    if result is not None:
        # Convert to correct types
        final_result = [list(c) for c in result]
        return final_result
    
    # Fallback: try sequential Hamiltonian cycle search
    def dfs_ham_cycle(avail_adj, start=0):
        """Find Hamiltonian cycle in graph defined by avail_adj."""
        path = [start]
        visited = {start}
        
        def dfs(curr):
            if len(path) == n:
                if start in avail_adj[curr]:
                    return True
                return False
            for nb in sorted(avail_adj[curr], key=lambda x: len(avail_adj[x])):
                if nb not in visited and nb != start:
                    visited.add(nb)
                    path.append(nb)
                    if dfs(nb):
                        return True
                    path.pop()
                    visited.remove(nb)
            return False
        
        if dfs(start):
            return path
        return None
    
    # Try to find 3 cycles sequentially
    best_result = None
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        used_edges = set()
        cycles = []
        success = True
        
        for _ in range(3):
            # Build available adjacency
            avail_adj = defaultdict(set)
            for i, (u, v) in enumerate(edges):
                if i not in used_edges:
                    avail_adj[u].add(v)
                    avail_adj[v].add(u)
            
            # Check degrees
            valid = True
            for v in range(n):
                if len(avail_adj[v]) < 2:
                    valid = False
                    break
            if not valid:
                success = False
                break
            
            # Find Hamiltonian cycle
            cycle = dfs_ham_cycle(avail_adj)
            if cycle is None:
                success = False
                break
            
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(get_edge_idx(u, v))
            
            cycles.append(cycle)
        
        if success and len(cycles) == 3:
            best_result = cycles
            break
    
    if best_result is not None:
        return [list(c) for c in best_result]
    
    # Last resort: find at least 1 cycle
    avail_adj = defaultdict(set)
    for u, v in edges:
        avail_adj[u].add(v)
        avail_adj[v].add(u)
    
    cycle = dfs_ham_cycle(avail_adj)
    if cycle is not None:
        return [list(cycle)]
    
    # Should not happen, but return a dummy
    return [list(range(n))]
# EVOLVE-BLOCK-END
