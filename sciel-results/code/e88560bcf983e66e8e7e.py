"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = time.time() + 85.0

def build_adjacency(edges):
    n = P['n']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamiltonian_cycle(adj, n, start, deadline):
    """Find a Hamiltonian cycle using DFS with strong pruning."""
    if time.time() > deadline:
        return None
    
    # Pre-check: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    sys.setrecursionlimit(10000)
    
    def dfs(current, path_len):
        if time.time() > deadline:
            return None
        if path_len == n:
            if start in adj[current]:
                return list(path)
            return None
        
        # Pruning: check if any unvisited vertex has < 2 available neighbors
        for v in range(n):
            if not visited[v] and v != start:
                count = 0
                for nb in adj[v]:
                    if not visited[nb]:
                        count += 1
                    elif nb == current:
                        count += 1
                    elif nb == start:
                        count += 1
                if count < 2:
                    return None
        
        # Special case: if only 2 unvisited vertices remain
        unvisited = [v for v in range(n) if not visited[v] and v != start]
        if len(unvisited) == 2:
            a, b = unvisited
            if a not in adj[b]:
                return None
            if not ((current in adj[a] and start in adj[b]) or (current in adj[b] and start in adj[a])):
                return None
        
        # Order candidates by fewest unvisited connections (most constrained first)
        candidates = []
        for nxt in adj[current]:
            if not visited[nxt]:
                unv_count = sum(1 for nb in adj[nxt] if not visited[nb] and nb != start)
                candidates.append((unv_count, nxt))
        candidates.sort()
        
        for _, nxt in candidates:
            visited[nxt] = True
            path.append(nxt)
            result = dfs(nxt, path_len + 1)
            if result is not None:
                return result
            path.pop()
            visited[nxt] = False
        return None
    
    return dfs(start, 1)

def construct():
    global DEADLINE
    n = P['n']
    edges = P['edges']
    
    adj = build_adjacency(edges)
    
    cycles = []
    used_edges = set()
    
    for attempt in range(3):
        if time.time() > DEADLINE:
            break
        
        remaining_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in used_edges:
                remaining_edges.append(e)
        
        rem_adj = build_adjacency(remaining_edges)
        valid = True
        for v in range(n):
            if len(rem_adj[v]) < 2:
                valid = False
                break
        
        if not valid:
            break
        
        cycle = None
        deadline = min(DEADLINE - 2.0, time.time() + 20.0)
        
        for start_v in range(n):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(rem_adj, n, start_v, deadline)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = (min(u, v), max(u, v))
            used_edges.add(key)
        
        cycles.append(cycle)
    
    if not cycles:
        return [[0, 1, 2]]
    
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
