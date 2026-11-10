"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_one_cycle(local_adj, n, used_edges, deadline, rng, start_vertex=0):
    """Find one Hamiltonian cycle starting from start_vertex using randomized DFS."""
    visited = {start_vertex}
    path = [start_vertex]
    
    def dfs(current):
        if time.time() > deadline:
            return None
        if len(path) == n:
            edge = (min(start_vertex, current), max(start_vertex, current))
            if start_vertex in local_adj[current] and edge not in used_edges:
                return path[:]
            return None
        neighbors = [x for x in local_adj[current] if x not in visited]
        # Filter by unused edges
        neighbors = [x for x in neighbors if (min(current, x), max(current, x)) not in used_edges]
        rng.shuffle(neighbors)
        for nxt in neighbors:
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.remove(nxt)
        return None
    
    return dfs(start_vertex)

def _find_hamiltonian_cycles(vertices, adj, k, deadline, seed=0):
    """
    Find k edge-disjoint Hamiltonian cycles on the given vertex set using randomized backtracking.
    """
    vertex_set = set(vertices)
    n = len(vertices)
    vlist = sorted(vertices)
    v_to_local = {v: i for i, v in enumerate(vlist)}
    
    # Build local adjacency
    local_adj = [set() for _ in range(n)]
    for u in vlist:
        for v in adj[u]:
            if v in vertex_set:
                local_adj[v_to_local[u]].add(v_to_local[v])
    
    # Total edges on this vertex set
    all_edges_local = set()
    for i in range(n):
        for j in local_adj[i]:
            if i < j:
                all_edges_local.add((i, j))
    
    if len(all_edges_local) < k * n:
        return None
    
    rng = random.Random(seed)
    used_edges = set()
    found = []
    
    for idx in range(k):
        if time.time() > deadline:
            return None
        # Try multiple starting vertices and restarts
        found_cycle = None
        for attempt in range(50):
            if time.time() > deadline:
                return None
            start_v = rng.randint(0, n - 1)
            cyc = _find_one_cycle(local_adj, n, used_edges, deadline, rng, start_v)
            if cyc is not None:
                found_cycle = cyc
                break
        
        if found_cycle is None:
            return None
        
        for i in range(n):
            u, v = found_cycle[i], found_cycle[(i + 1) % n]
            used_edges.add((min(u, v), max(u, v)))
        found.append([vlist[v] for v in found_cycle])
    
    return found

def construct():
    deadline = time.time() + 80
    
    adj = _build_adj()
    n = P['n']
    full_set = list(range(n))
    
    total_edges = len(P['edges'])
    max_possible = total_edges // n
    
    best_result = None
    
    for k in range(max_possible, 0, -1):
        # Try multiple seeds to escape local minima
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = _find_hamiltonian_cycles(full_set, adj, k, deadline, seed=seed)
            if result is not None:
                cycles = []
                for cyc in result:
                    cycles.append([int(v) for v in cyc])
                return cycles
        
        # If we couldn't find k cycles, try k-1
        # (handled by the outer loop)
    
    # Fallback: find any triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in edge_set:
                continue
            for c in range(b + 1, n):
                if (b, c) in edge_set and (a, c) in edge_set:
                    return [[a, b, c]]
    
    return [[0, 1, 2]]
# EVOLVE-BLOCK-END
