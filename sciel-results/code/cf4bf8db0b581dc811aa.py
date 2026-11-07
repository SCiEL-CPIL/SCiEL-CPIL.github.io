"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_set(cycle):
    """Return set of frozenset edges used by a cycle."""
    es = set()
    k = len(cycle)
    for i in range(k):
        a, b = cycle[i], cycle[(i + 1) % k]
        es.add(frozenset((a, b)))
    return es

def _find_hamiltonian_cycle(adj, available_edges, n, deadline, rng):
    """
    Find a Hamiltonian cycle on all n vertices using only edges in available_edges.
    Uses Warnsdorff-guided backtracking.
    Returns list of vertices or None.
    """
    # Precompute degree in available graph
    deg = [0] * n
    for i in range(n):
        deg[i] = len(available_edges[i])
    
    # Check if all degrees >= 2
    if any(deg[i] < 2 for i in range(n)):
        return None
    
    # Start from vertex with smallest degree for determinism-ish
    start = min(range(n), key=lambda i: (deg[i], i))
    
    best_cycle = None
    max_backtracks = 500000
    
    def backtrack(path, used_verts, used_edges):
        nonlocal best_cycle, max_backtracks
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check closing edge
            if (path[-1], path[0]) in used_edges or (path[0], path[-1]) in used_edges:
                # Actually check adjacency
                if path[0] in available_edges[path[-1]]:
                    best_cycle = list(path)
                    return best_cycle
            return None
        
        current = path[-1]
        candidates = [v for v in available_edges[current] if v not in used_verts]
        
        if not candidates:
            return None
        
        # Warnsdorff: sort by number of unvisited neighbors
        def warnsdorff_score(v):
            return sum(1 for u in available_edges[v] if u not in used_verts and u != current)
        
        candidates.sort(key=warnsdorff_score)
        
        # Add some randomness
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        
        for v in candidates:
            used_verts.add(v)
            path.append(v)
            result = backtrack(path, used_verts, used_edges)
            if result is not None:
                return result
            path.pop()
            used_verts.discard(v)
            max_backtracks -= 1
            if max_backtracks <= 0:
                return None
        return None
    
    path = [start]
    used_verts = {start}
    used_edges = set()
    result = backtrack(path, used_verts, used_edges)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # All vertices
    all_verts = set(range(n))
    
    # Start with all edges available
    available = [set(adj[i]) for i in range(n)]
    
    rng = random.Random(42)
    
    cycles_found = []
    used_edges_global = set()
    
    # Try to find up to 3 Hamiltonian cycles (70 edges / 21 = ~3.3)
    max_cycles = 3
    
    for target_cycle in range(max_cycles):
        if time.time() > deadline:
            break
        
        # Try multiple restarts with different random seeds
        found = False
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Create a fresh copy of available edges (excluding already used)
            avail_copy = [set() for i in range(n)]
            for i in range(n):
                for j in available[i]:
                    if frozenset((i, j)) not in used_edges_global:
                        avail_copy[i].add(j)
                # Ensure symmetry
                for j in avail_copy[i]:
                    if i not in avail_copy[j]:
                        avail_copy[j].add(i)
        
            cycle = _find_hamiltonian_cycle(avail_copy, avail_copy, n, deadline, rng)
            if cycle is not None:
                # Validate the cycle
                valid = True
                k = len(cycle)
                if k < 3 or len(set(cycle)) != k:
                    valid = False
                if valid:
                    for i in range(k):
                        a, b = cycle[i], cycle[(i + 1) % k]
                        if b not in avail_copy[a]:
                            valid = False
                            break
                        # Check edge not already used
                        if frozenset((a, b)) in used_edges_global:
                            valid = False
                            break
                if valid:
                    # Mark edges as used
                    for i in range(k):
                        a, b = cycle[i], cycle[(i + 1) % k]
                        used_edges_global.add(frozenset((a, b)))
                    cycles_found.append(list(cycle))
                    found = True
                    break
        
        if not found:
            break
    
    # If we didn't find any cycles (shouldn't happen), return a fallback
    if not cycles_found:
        # Return a single Hamiltonian cycle if we can find one
        avail_copy = [set(adj[i]) for i in range(n)]
        cycle = _find_hamiltonian_cycle(avail_copy, avail_copy, n, deadline, rng)
        if cycle is not None:
            cycles_found.append(list(cycle))
        else:
            # Absolute fallback: return empty? No, must be nonempty.
            # Try a simple cycle
            return [[0, 1, 2]]
    
    # Ensure all cycles use the same vertex set (they should, since Hamiltonian)
    # Convert to proper types
    result = []
    for c in cycles_found:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
