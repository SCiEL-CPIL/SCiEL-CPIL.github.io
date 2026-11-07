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
    deadline = time.time() + 85  # 85 seconds to be safe
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # All vertices
    all_vertices = list(range(n))
    
    # Start with no used edges
    used_edges = set()
    cycles = []
    
    # Try to find as many edge-disjoint Hamiltonian cycles as possible
    # Greedy approach: find a cycle, add it, remove its edges, repeat
    
    remaining_vertices = set(all_vertices)
    
    for i in range(10):  # Max 10 cycles
        if time.time() > deadline:
            break
        
        # Find a Hamiltonian cycle on all vertices
        cycle = _find_ham_cycle(all_vertices, adj, used_edges, deadline)
        
        if cycle is None:
            break
        
        # Add edges of this cycle to used_edges
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            used_edges.add(e)
        
        cycles.append(cycle)
    
    # Convert to proper types
    result = [list(c) for c in cycles]
    return result
# EVOLVE-BLOCK-END
