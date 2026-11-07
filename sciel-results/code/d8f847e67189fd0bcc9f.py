"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, deadline, start_override=None):
    """
    Find a Hamiltonian cycle in the graph defined by adj using DFS with Warnsdorff heuristic.
    Returns a list of vertices representing the cycle, or None if not found.
    """
    if start_override is not None:
        start = start_override
    else:
        start = random.randint(0, n - 1)
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    sys.setrecursionlimit(10000)
    
    def dfs(current):
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            if start in adj[current]:
                return True
            return False
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        # Warnsdorff heuristic: prefer neighbors with fewer unvisited options
        # This helps avoid dead ends
        def heuristic(v):
            unvisited_count = sum(1 for w in adj[v] if not visited[w])
            return unvisited_count
        
        neighbors.sort(key=heuristic)
        
        # Shuffle with some randomness to explore different paths
        # But keep the heuristic ordering as primary
        if len(neighbors) > 1:
            # With some probability, try a non-greedy choice to escape local optima
            if random.random() < 0.1:
                neighbors = neighbors[::-1]  # reverse order sometimes
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            if dfs(v):
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs(start):
        return path
    return None

def _edges_from_cycle(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if u > v:
            u, v = v, u
        edges.add((u, v))
    return edges

def _remove_edges(adj, edges_to_remove):
    """Remove edges from adjacency sets."""
    for u, v in edges_to_remove:
        adj[u].discard(v)
        adj[v].discard(u)
    return adj

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline guard: 85 seconds to be safe within 90s limit
    deadline = time.time() + 85.0
    
    best_cycles = []
    
    # Try multiple random restarts to find a good set of edge-disjoint Hamiltonian cycles
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Build fresh adjacency from original edges
        adj = _build_adj(n, edges)
        cycles = []
        success = True
        
        for cycle_idx in range(10):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Try to find a Hamiltonian cycle with different starting vertices
            cycle = None
            found = False
            
            # Try several starting vertices
            start_vertices = list(range(n))
            random.shuffle(start_vertices)
            
            for start in start_vertices:
                if time.time() > deadline - 5:
                    found = False
                    break
                
                candidate = _find_ham_cycle(adj, n, deadline, start_override=start)
                if candidate is not None:
                    cycle = candidate
                    found = True
                    break
            
            if not found or cycle is None:
                success = False
                break
            
            cycles.append(list(cycle))
            
            # Remove the edges used in this cycle
            edge_set = _edges_from_cycle(cycle)
            _remove_edges(adj, edge_set)
        
        if len(cycles) > len(best_cycles):
            best_cycles = [list(c) for c in cycles]
        
        # If we found a lot of cycles, we can stop early
        if len(best_cycles) >= 7:
            break
    
    # Convert to proper format: list of lists of Python ints
    result = []
    for cycle in best_cycles:
        # Ensure all elements are Python ints
        clean_cycle = [int(v) for v in cycle]
        result.append(clean_cycle)
    
    # Validate the result
    if not result:
        # Fallback: return a single known Hamiltonian cycle if we found one
        # Use a simple greedy approach
        adj = _build_adj(n, edges)
        cycle = _find_ham_cycle(adj, n, deadline, start_override=0)
        if cycle is not None:
            result = [[int(v) for v in cycle]]
    
    # Final validation: check that all cycles use the same vertex set
    if result:
        vertex_set = set(result[0])
        for cycle in result:
            if set(cycle) != vertex_set:
                result = [[int(v) for v in result[0]]]  # Keep only first valid cycle
                break
    
    # Check edge-disjointness
    if len(result) > 1:
        used_edges = set()
        valid = []
        for cycle in result:
            edge_set = _edges_from_cycle(cycle)
            if edge_set & used_edges:
                break  # Not edge-disjoint
            used_edges |= edge_set
            valid.append(cycle)
        result = valid
    
    return result

# EVOLVE-BLOCK-END
