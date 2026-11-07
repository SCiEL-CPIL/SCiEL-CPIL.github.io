"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, deadline, max_depth=5000):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking."""
    vertices = list(vertices)
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # State: (current_vertex, visited_set, path)
    # We use iterative DFS with explicit stack to avoid recursion limit
    # But for 21 vertices, recursion with sys.setrecursionlimit might work
    # Let's use iterative approach for safety
    
    # Actually, let's try a recursive approach with memoization of visited states
    # But the state space is too large for memoization.
    # Let's use a simple DFS with backtracking.
    
    # To make it faster, we can use a heuristic: always try to visit unvisited vertices
    # that have the fewest unvisited neighbors (Warnsdorff's rule for Hamiltonian paths)
    
    def dfs(current, visited, path):
        if len(visited) == n:
            # Check if we can close the cycle
            if start in adj[current] and (min(current, start), max(current, start)) not in used_edges:
                return path
            return None
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj[current]:
            if nb in vset and nb not in visited:
                # Check if edge (current, nb) is not used
                e = (min(current, nb), max(current, nb))
                if e not in used_edges:
                    unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree in remaining graph (Warnsdorff's heuristic)
        unvisited_neighbors.sort(key=lambda x: len([y for y in adj[x] if y in vset and y not in visited and (min(x, y), max(x, y)) not in used_edges]))
        
        for nb in unvisited_neighbors:
            visited.add(nb)
            path.append(nb)
            e = (min(current, nb), max(current, nb))
            used_edges.add(e)
            
            result = dfs(nb, visited, path)
            if result is not None:
                return result
            
            # Backtrack
            path.pop()
            visited.discard(nb)
            used_edges.discard(e)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(start, visited, path)

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    # The graph is C_21(1, 2, 5).
    # We can construct 3 edge-disjoint Hamiltonian cycles using steps 1, 2, and 5.
    
    cycles = []
    
    # Cycle 1: Step 1
    c1 = list(range(n))
    # Verify c1 is a valid cycle in the graph
    valid = True
    for i in range(n):
        u = c1[i]
        v = c1[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            valid = False
            break
    if valid:
        cycles.append(c1)
        
    # Cycle 2: Step 2
    c2 = [(i * 2) % n for i in range(n)]
    valid = True
    for i in range(n):
        u = c2[i]
        v = c2[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            valid = False
            break
    if valid:
        cycles.append(c2)
        
    # Cycle 3: Step 5
    c3 = [(i * 5) % n for i in range(n)]
    valid = True
    for i in range(n):
        u = c3[i]
        v = c3[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            valid = False
            break
    if valid:
        cycles.append(c3)
        
    # Verify edge-disjointness
    used_edges = set()
    final_cycles = []
    for c in cycles:
        is_disjoint = True
        for i in range(len(c)):
            u = c[i]
            v = c[(i + 1) % len(c)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                is_disjoint = False
                break
        if is_disjoint:
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            final_cycles.append(c)
            
    return [list(c) for c in final_cycles]
# EVOLVE-BLOCK-END
