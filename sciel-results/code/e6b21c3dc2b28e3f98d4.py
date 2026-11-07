"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, used_edges, start, deadline, rng):
    """Warnsdorff-guided DFS to find a Hamiltonian cycle in the remaining graph."""
    n = len(adj)
    path = [start]
    visited = [False] * n
    visited[start] = True
    # Precompute adjacency as sorted lists for faster iteration
    adj_list = [sorted(adj[v]) for v in range(n)]
    
    def dfs(u):
        if time.time() > deadline:
            return False
        if len(path) == n:
            return u in adj[start] and (min(u, start), max(u, start)) not in used_edges
        # Warnsdorff: sort neighbors by degree in remaining graph
        neighbors = [v for v in adj_list[u] if not visited[v] and (min(u, v), max(u, v)) not in used_edges]
        if not neighbors:
            return False
        # Sort by number of available moves (Warnsdorff's rule)
        def warnsdorff_score(v):
            avail = 0
            for w in adj_list[v]:
                if not visited[w] and (min(v, w), max(v, w)) not in used_edges:
                    avail += 1
            return avail
        neighbors.sort(key=warnsdorff_score)
        # Add small random perturbation for diversity
        if rng.random() < 0.15 and len(neighbors) > 1:
            idx = rng.randint(0, len(neighbors) - 1)
            neighbors[idx], neighbors[0] = neighbors[0], neighbors[idx]
        
        for v in neighbors:
            path.append(v)
            visited[v] = True
            if dfs(v):
                return True
            path.pop()
            visited[v] = False
        return False
    
    if dfs(start):
        return list(path)
    return None

def _cycle_edges(cycle):
    """Get the set of edges used by a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add((min(u, v), max(u, v)))
    return edges

def _try_pack(adj, deadline, rng):
    """Try to pack edge-disjoint Hamiltonian cycles."""
    used_edges = set()
    cycles = []
    
    while len(cycles) < 10:
        if time.time() > deadline:
            break
        # Find a start vertex with available edges
        candidates = [v for v in range(P['n']) if adj[v] and any((min(v, w), max(v, w)) not in used_edges for w in adj[v])]
        if not candidates:
            break
        # Shuffle for diversity
        rng.shuffle(candidates)
        found = False
        for start in candidates:
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng)
            if cycle:
                ce = _cycle_edges(cycle)
                if not ce & used_edges:
                    used_edges.update(ce)
                    cycles.append(cycle)
                    found = True
                    break
        if not found:
            break
    
    return cycles

def _local_search(cycles, adj, deadline, rng):
    """Try to improve packing by swapping edges between cycles to free up space for more cycles."""
    if len(cycles) < 2:
        return cycles
    
    n = P['n']
    best_cycles = list(cycles)
    best_count = len(cycles)
    
    # Try to find alternative Hamiltonian cycles for existing cycles that use fewer "conflicting" edges
    for _ in range(20):  # Limited local search iterations
        if time.time() > deadline:
            break
        
        # For each cycle, try to find an alternative that avoids some edges
        improved = False
        for i in range(len(best_cycles)):
            if time.time() > deadline:
                break
            
            other_edges = set()
            for j in range(len(best_cycles)):
                if j != i:
                    other_edges |= _cycle_edges(best_cycles[j])
            
            # Find edges in this cycle that are also in other cycles' potential paths
            # Try to find an alternative cycle that avoids certain edges
            cycle_edges = _cycle_edges(best_cycles[i])
            # Pick a random edge to try to avoid
            edge_to_avoid = rng.choice(list(cycle_edges)) if cycle_edges else None
            
            if edge_to_avoid:
                used = set()
                for j in range(len(best_cycles)):
                    if j != i:
                        used |= _cycle_edges(best_cycles[j])
                
                # Try to find a new cycle avoiding edge_to_avoid
                start = best_cycles[i][0]
                alt_cycle = _find_ham_cycle(adj, used | {edge_to_avoid}, start, deadline, rng)
                if alt_cycle:
                    alt_ce = _cycle_edges(alt_cycle)
                    # Check it doesn't conflict with other cycles
                    if not (alt_ce & used):
                        best_cycles[i] = alt_cycle
                        improved = True
        
        if not improved:
            break
    
    # Try to add more cycles after local search
    used_edges = set()
    for c in best_cycles:
        used_edges |= _cycle_edges(c)
    
    while len(best_cycles) < 10:
        if time.time() > deadline:
            break
        candidates = [v for v in range(n) if adj[v] and any((min(v, w), max(v, w)) not in used_edges for w in adj[v])]
        if not candidates:
            break
        rng.shuffle(candidates)
        found = False
        for start in candidates:
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng)
            if cycle:
                ce = _cycle_edges(cycle)
                if not ce & used_edges:
                    used_edges.update(ce)
                    best_cycles.append(cycle)
                    found = True
                    break
        if not found:
            break
    
    return best_cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    adj = _build_adj()
    
    best = [list(range(P['n']))]  # Fallback: single Hamiltonian cycle
    best_count = 1
    
    # Multiple random restarts with different seeds and strategies
    for attempt in range(500):
        if time.time() > deadline:
            break
        rng = random.Random(attempt * 7919 + 13)
        cycles = _try_pack(adj, deadline, rng)
        
        # Apply local search to improve
        if len(cycles) >= 2:
            cycles = _local_search(cycles, adj, deadline, rng)
        
        if len(cycles) > best_count:
            best = cycles
            best_count = len(cycles)
            if best_count >= 4:
                break  # Likely near theoretical max
    
    # Convert to proper types
    result = []
    for c in best:
        result.append([int(v) for v in c])
    return result
# EVOLVE-BLOCK-END
