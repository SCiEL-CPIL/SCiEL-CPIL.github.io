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

def _cycle_edges(cycle):
    """Get the set of edges used by a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add((min(u, v), max(u, v)))
    return edges

def _find_ham_cycle(adj, used_edges, start, deadline, rng, adj_list):
    """Warnsdorff-guided DFS with constraint propagation to find a Hamiltonian cycle."""
    n = len(adj)
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute available neighbors for each vertex given used_edges
    # avail_neighbors[v] = list of neighbors w such that edge (v,w) not in used_edges
    avail_neighbors = [[] for _ in range(n)]
    for v in range(n):
        for w in adj_list[v]:
            e = (min(v, w), max(v, w))
            if e not in used_edges:
                avail_neighbors[v].append(w)
    
    def dfs(u, depth):
        if time.time() > deadline:
            return False
        if depth == n:
            # Check if we can close the cycle
            e = (min(u, start), max(u, start))
            return (min(u, start), max(u, start)) not in used_edges and u in adj[start]
        
        # Get unvisited neighbors with available edges
        candidates = []
        for v in avail_neighbors[u]:
            if not visited[v]:
                candidates.append(v)
        
        if not candidates:
            return False
        
        # Warnsdorff: sort by number of available moves
        def warnsdorff_score(v):
            count = 0
            for w in avail_neighbors[v]:
                if not visited[w]:
                    count += 1
            return count
        
        candidates.sort(key=warnsdorff_score)
        
        # Add small random perturbation for diversity
        if rng.random() < 0.1 and len(candidates) > 1:
            idx = rng.randint(0, len(candidates) - 1)
            candidates[idx], candidates[0] = candidates[0], candidates[idx]
        
        for v in candidates:
            path.append(v)
            visited[v] = True
            if dfs(v, depth + 1):
                return True
            path.pop()
            visited[v] = False
        return False
    
    if dfs(start, 1):
        return list(path)
    return None

def _try_pack(adj, deadline, rng):
    """Try to pack edge-disjoint Hamiltonian cycles."""
    n = P['n']
    adj_list = [sorted(adj[v]) for v in range(n)]
    used_edges = set()
    cycles = []
    
    while len(cycles) < 3:
        if time.time() > deadline:
            break
        
        # Find a start vertex with available edges
        candidates = []
        for v in range(n):
            has_avail = False
            for w in adj[v]:
                if (min(v, w), max(v, w)) not in used_edges:
                    has_avail = True
                    break
            if has_avail:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Shuffle for diversity
        rng.shuffle(candidates)
        found = False
        for start in candidates:
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng, adj_list)
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

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    adj = _build_adj()
    n = P['n']
    
    best = [list(range(n))]  # Fallback: single Hamiltonian cycle
    best_count = 1
    
    # Multiple random restarts with different seeds
    for attempt in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(attempt * 7919 + 13)
        cycles = _try_pack(adj, deadline, rng)
        
        if len(cycles) > best_count:
            best = cycles
            best_count = len(cycles)
            if best_count >= 3:
                break  # Theoretical maximum reached
    
    # Convert to proper types
    result = []
    for c in best:
        result.append([int(v) for v in c])
    return result
# EVOLVE-BLOCK-END
