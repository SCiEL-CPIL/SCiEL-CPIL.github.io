"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, deadline, rng):
    """Find a Hamiltonian cycle in the graph defined by adj (list of sets)."""
    # Try multiple starting vertices
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            return None
        if len(adj[start]) < 2:
            continue
        
        # Backtracking with Warnsdorff heuristic
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() > deadline:
                return False
            if len(path) == n:
                # Check if we can close the cycle
                if start in adj[path[-1]]:
                    return True
                return False
            
            # Get unvisited neighbors of current vertex
            cur = path[-1]
            neighbors = [v for v in adj[cur] if v not in visited]
            
            if not neighbors:
                return False
            
            # Warnsdorff: sort by degree in remaining graph (fewest unvisited neighbors first)
            def heuristic(v):
                return len([w for w in adj[v] if w not in visited])
            
            neighbors.sort(key=heuristic)
            
            for v in neighbors:
                path.append(v)
                visited.add(v)
                if backtrack():
                    return True
                path.pop()
                visited.discard(v)
            
            return False
        
        if backtrack():
            return path
    
    return None

def _find_ham_cycle_with_restarts(adj, n, deadline, rng, max_restarts=50):
    """Try multiple times with different random seeds/orderings."""
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        # Shuffle adjacency lists to get different search orders
        adj_shuffled = {v: list(u) for v, u in adj.items()}
        for v in adj_shuffled:
            rng.shuffle(adj_shuffled[v])
        adj_shuffled = {v: set(u) for v, u in adj_shuffled.items()}
        
        cycle = _find_ham_cycle(adj_shuffled, n, deadline, rng)
        if cycle:
            return cycle
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline: use 85 seconds to be safe (90s wall clock limit)
    start_time = time.time()
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Build initial adjacency
    adj = _build_adj(edges)
    
    cycles = []
    used_edges = set()
    
    # Try to find up to 10 edge-disjoint Hamiltonian cycles
    max_cycles = 10
    
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        # Build adjacency excluding used edges
        adj_remaining = defaultdict(set)
        for u, v in edges:
            if (u, v) not in used_edges and (v, u) not in used_edges:
                adj_remaining[u].add(v)
                adj_remaining[v].add(u)
        
        # Check if all vertices have degree >= 2 in remaining graph
        valid = True
        for v in range(n):
            if v not in adj_remaining or len(adj_remaining[v]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Try to find a Hamiltonian cycle
        cycle = _find_ham_cycle_with_restarts(adj_remaining, n, deadline, rng, max_restarts=30)
        
        if cycle is None:
            break
        
        # Convert cycle to edge set
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_edges.add((min(u, v), max(u, v)))
        
        # Check edge-disjointness (should be guaranteed by construction, but verify)
        conflict = False
        for e in cycle_edges:
            if e in used_edges:
                conflict = True
                break
        if conflict:
            break
        
        # Add to used edges and cycles
        used_edges.update(cycle_edges)
        cycles.append([int(v) for v in cycle])
    
    # Ensure we have at least one cycle (the trivial one if possible)
    if not cycles:
        # Fallback: try to find any Hamiltonian cycle in the original graph
        adj_orig = _build_adj(edges)
        cycle = _find_ham_cycle_with_restarts(adj_orig, n, deadline, rng, max_restarts=100)
        if cycle:
            cycles.append([int(v) for v in cycle])
        else:
            # Last resort: return a trivial cycle if graph has one
            # This shouldn't happen for this graph
            cycles.append([0, 1, 2])  # placeholder, will be invalid but better than empty
    
    # Convert to proper Python ints and lists
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
