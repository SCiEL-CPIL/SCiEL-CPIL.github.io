"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _find_hamiltonian_cycle(adj, used_edges, n, start=0):
    """Backtracking search for a Hamiltonian cycle avoiding used_edges."""
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    def backtrack():
        if len(path) == n:
            # Check if we can return to start
            if _edge_key(path[-1], start) not in used_edges:
                return True
            return False
        
        last = path[-1]
        # Try neighbors in a heuristic order: prefer those with fewer remaining options
        neighbors = sorted(adj[last], key=lambda x: (
            0 if _edge_key(last, x) in used_edges else 1,
            len(adj[x])
        ))
        
        for nb in neighbors:
            if visited[nb]:
                continue
            ek = _edge_key(last, nb)
            if ek in used_edges:
                continue
            # Pruning: if nb is not start and we're not at the last step,
            # nb must have at least one unvisited neighbor (or be the final step to connect back)
            if len(path) < n - 1:
                has_unvisited_neighbor = False
                for nn in adj[nb]:
                    if not visited[nn] and _edge_key(nb, nn) not in used_edges:
                        has_unvisited_neighbor = True
                        break
                if not has_unvisited_neighbor:
                    continue
            
            visited[nb] = True
            path.append(nb)
            used_edges.add(ek)
            
            if backtrack():
                return True
            
            used_edges.discard(ek)
            path.pop()
            visited[nb] = False
        
        return False
    
    if backtrack():
        return path[:]
    return None

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Deadline guard
    deadline = time.time() + 80  # 80s safety margin
    
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        cycle = _find_hamiltonian_cycle(adj, used_edges, n)
        if cycle is None:
            break
        
        # Add all edges of this cycle to used_edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(cycle)
    
    # Ensure at least one cycle; if none found, return a single Hamiltonian cycle
    if not cycles:
        # Fallback: find one without edge constraints
        used_edges = set()
        cycle = _find_hamiltonian_cycle(adj, used_edges, n)
        if cycle:
            cycles = [cycle]
        else:
            # Absolute fallback
            cycles = [list(range(n))]
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append(tuple(int(v) for v in c))
    
    return result
# EVOLVE-BLOCK-END
