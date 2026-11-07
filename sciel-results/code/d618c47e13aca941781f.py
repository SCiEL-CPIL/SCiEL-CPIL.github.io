"""Model-and-optimize: search vertex sets, fit edge-disjoint Hamiltonian cycles via randomized greedy."""
import itertools
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE_MARGIN = 10  # seconds before kill to stop

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_one_ham_cycle(S_list, adj_S, used_edges, rng, deadline):
    """Find one Hamiltonian cycle on S_list avoiding used_edges using backtracking."""
    k = len(S_list)
    if k < 3:
        return None
    start = S_list[0]
    
    visited = {start}
    path = [start]
    local_edges = set()
    
    def dfs():
        if time.time() > deadline:
            return False
        if len(path) == k:
            last = path[-1]
            close_edge = frozenset([last, start])
            if close_edge not in used_edges and close_edge not in local_edges:
                if start in adj_S[last]:
                    return True
            return False
        
        cur = path[-1]
        neighbors = list(adj_S[cur])
        
        # Heuristic: Sort by fewest unvisited neighbors (Warnsdorff)
        def heuristic(nb):
            if nb in visited:
                return float('inf')
            unvisited_neighbors = 0
            for x in adj_S[nb]:
                if x not in visited and x != cur:
                    unvisited_neighbors += 1
            return unvisited_neighbors
            
        neighbors.sort(key=heuristic)
        
        for nb in neighbors:
            if nb in visited:
                continue
            e = frozenset([cur, nb])
            if e in used_edges or e in local_edges:
                continue
            
            # Try adding nb
            visited.add(nb)
            local_edges.add(e)
            path.append(nb)
            
            if dfs():
                return True
            
            # Backtrack
            path.pop()
            local_edges.remove(e)
            visited.remove(nb)
            
        return False

    if dfs():
        return path
    return None

def _find_cycles_for_set(S_list, adj, target_count, deadline, rng):
    """Find up to target_count edge-disjoint Hamiltonian cycles on S_list."""
    S_set = frozenset(S_list)
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
    
    # Check degree: each vertex must have degree >= 2*target_count in adj_S
    # Actually each vertex needs at least 2 edges per cycle, so degree >= 2*target_count
    for v in S_list:
        if len(adj_S[v]) < 2 * target_count:
            return None  # Not enough edges
    
    used_edges = set()
    cycles = []
    for i in range(target_count):
        if time.time() > deadline:
            break
        cycle = _find_one_ham_cycle(S_list, adj_S, used_edges, rng, deadline)
        if cycle is None:
            break
        # Add edges to used
        n = len(cycle)
        for j in range(n):
            u = cycle[j]
            v = cycle[(j + 1) % n]
            used_edges.add(frozenset([u, v]))
        cycles.append(cycle)
    
    if len(cycles) == target_count:
        return cycles
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    rng = random.Random(42)
    
    adj = _build_adjacency()
    n = P['n']
    all_vertices = list(range(n))
    
    best_cycles = None
    best_count = 0
    
    # Try vertex set sizes from large to small
    # Full set (21) is the only one that can support 3 cycles in this 6-regular graph
    sizes_to_try = [21, 18, 17, 15, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    
    for s in sizes_to_try:
        if time.time() > deadline - 5:
            break
        
        max_possible = min(10, 70 // s)
        if max_possible <= best_count:
            continue
        
        # If we already found the theoretical max (3), stop
        if best_count >= 3:
            break
        
        # Generate candidate vertex sets
        if s <= 5:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s <= 7:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s <= 9:
            candidates = list(itertools.combinations(all_vertices, s))[:5000]
        elif s <= 12:
            candidates = list(itertools.combinations(all_vertices, s))[:2000]
        else:
            candidates = list(itertools.combinations(all_vertices, s))[:500]
        
        # Shuffle to randomize order
        rng.shuffle(candidates)
        
        for candidate in candidates:
            if time.time() > deadline - 3:
                break
            
            S_list = list(candidate)
            cycles = _find_cycles_for_set(S_list, adj, max_possible, deadline, rng)
            
            if cycles is not None:
                best_cycles = [list(c) for c in cycles]
                best_count = len(cycles)
                
                if best_count >= max_possible and max_possible >= 10:
                    break
        
        if best_count >= 10:
            break
    
    # Fallback: find any single cycle
    if best_cycles is None or len(best_cycles) == 0:
        # Find a triangle
        edge_set = set()
        for u, v in P['edges']:
            edge_set.add(frozenset([u, v]))
        
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    if frozenset([w, u]) in edge_set and frozenset([w, v]) in edge_set:
                        best_cycles = [[u, v, w]]
                        best_count = 1
                        break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        # Absolute fallback
        return [list(range(P['n']))]
    
    # Validate and convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result
