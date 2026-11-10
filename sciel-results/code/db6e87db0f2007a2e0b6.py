"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = set(range(N))
EDGE_SET = set()
ADJ = defaultdict(set)
for u, v in P['edges']:
    edge = (min(u, v), max(u, v))
    EDGE_SET.add(edge)
    ADJ[u].add(v)
    ADJ[v].add(u)

def edge_key(u, v):
    return (min(u, v), max(u, v))

def find_hamiltonian_cycle(used_edges, start, deadline):
    """Find a Hamiltonian cycle in the remaining graph using backtracking."""
    if time.time() > deadline:
        return None
    
    n = N
    # Build adjacency for remaining graph
    rem_adj = defaultdict(set)
    for u in range(n):
        for v in ADJ[u]:
            ek = edge_key(u, v)
            if ek not in used_edges:
                rem_adj[u].add(v)
    
    # Check if start has degree >= 2
    if len(rem_adj[start]) < 2:
        return None
    
    # Quick connectivity check: is the remaining graph connected?
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in rem_adj[node]:
            if nb not in visited:
                stack.append(nb)
    if len(visited) != n:
        return None
    
    # Check all vertices have degree >= 2
    for v in range(n):
        if len(rem_adj[v]) < 2:
            return None
    
    # Backtracking search for Hamiltonian cycle
    path = [start]
    used_in_path = {start}
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex connects to start
            last = path[-1]
            if start in rem_adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        # Get neighbors not in path, sorted for determinism/pruning
        candidates = [v for v in sorted(rem_adj[last]) if v not in used_in_path]
        
        # Pruning: if we can't complete the path, skip
        for v in candidates:
            # Check if v has enough remaining degree to be useful
            # v will be used, so it needs at least 1 other unvisited neighbor (unless it's the last)
            if len(path) == n - 1:
                # v must connect back to start
                if start not in rem_adj[v]:
                    continue
            else:
                # v needs at least one unvisited neighbor besides last
                unvisited_neighbors = [x for x in rem_adj[v] if x not in used_in_path and x != last]
                if not unvisited_neighbors:
                    continue
            
            path.append(v)
            used_in_path.add(v)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            used_in_path.remove(v)
        
        return None
    
    result = backtrack()
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_result = [list(range(N))]  # fallback: one valid cycle
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try different starting vertices and orderings
        used_edges = set()
        cycles = []
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            cycle = find_hamiltonian_cycle(used_edges, i, deadline)
            if cycle is not None:
                # Mark edges as used
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    used_edges.add(edge_key(u, v))
                cycles.append(cycle)
            
            if len(cycles) >= 3:
                break
        
        if len(cycles) > len(best_result):
            best_result = cycles
        
        if len(best_result) >= 3:
            break
    
    # Validate and return
    if not best_result:
        return [list(range(N))]
    
    # Ensure types are correct
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
