"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _cycle_edges(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges

def _verify_cycle(cycle, n, edge_set):
    """Verify cycle is valid: all distinct vertices, all edges in host graph."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            return False
    return True

def construct():
    n = P['n']
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
    
    # The graph is C_21(1,2,5): vertex i connects to i±1, i±2, i±5 (mod 21)
    # Construct 3 edge-disjoint Hamiltonian cycles using steps 1, 2, and 5.
    # Each step-s cycle uses only edges of type (i, i+s) which are distinct across s in {1,2,5}.
    # Max possible = floor(75/21) = 3, so this is optimal.
    
    cycles = []
    used_edges = set()
    
    for step in [1, 2, 5]:
        # Build cycle: 0, step, 2*step, ..., 20*step (mod 21)
        cycle = []
        for i in range(n):
            cycle.append((i * step) % n)
        
        # Verify this is a valid Hamiltonian cycle
        if not _verify_cycle(cycle, n, host_edges):
            continue
        
        # Check edge-disjointness
        ce = _cycle_edges(cycle)
        if ce & used_edges:
            continue
        
        used_edges |= ce
        cycles.append(cycle)
    
    # Fallback: if structural approach fails, try greedy DFS
    if len(cycles) < 1:
        from collections import defaultdict
        import random
        start_time = time.time()
        deadline = start_time + 85
        rng = random.Random(42)
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            adj = defaultdict(set)
            for u, v in P['edges']:
                adj[u].add(v)
                adj[v].add(u)
            
            found_cycle = None
            path = [0]
            visited = {0}
            
            def dfs(current, path, visited):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    if 0 in adj.get(current, set()):
                        return list(path)
                    return None
                neighbors = [x for x in adj[current] if x not in visited]
                neighbors.sort(key=lambda x: len(adj[x] - visited))
                for nxt in neighbors:
                    visited.add(nxt)
                    path.append(nxt)
                    result = dfs(nxt, path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nxt)
                return None
            
            found_cycle = dfs(0, path, visited)
            if found_cycle is not None and _verify_cycle(found_cycle, n, host_edges):
                cycles = [found_cycle]
                break
    
    # Ensure proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    if not result:
        result = [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
