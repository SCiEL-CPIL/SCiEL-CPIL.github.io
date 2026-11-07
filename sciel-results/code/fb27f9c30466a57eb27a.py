"""Rewritten with structural change: reverse search + local optimization."""
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    """Return diameter or -1 if disconnected."""
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d

def _is_triangle_free(n, edges):
    """Check triangle-free by verifying bipartiteness (since we only add opposite-parity edges)."""
    adj = _build_adj(n, edges)
    colors = [-1] * n
    for i in range(n):
        if colors[i] == -1:
            colors[i] = 0
            q = [i]
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        return False
    return True

def _valid_solution(n, base_edges, added_edges):
    """Check if the solution is valid: triangle-free, connected, diameter <= 4."""
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False  # duplicate
        edge_set.add((a, b))
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    diam = _bfs_diameter(n, adj)
    return diam != -1 and diam <= 4

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # Generate all valid candidate edges (opposite parity, not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:  # opposite parity ensures bipartite
                candidates.append((u, v))
    
    deadline = time.time() + 80
    
    # Strategy 1: Try known good structures first
    # Structure: connect vertices in a pattern that creates diameter 4
    # Idea: partition into 6 groups of 4, connect group centers
    # Or: use a "star-like" bipartite structure
    
    # Let's try several candidate solutions and pick the best valid one
    
    # Solution A: 5 edges creating a "ladder" shortcut
    # 0-12, 6-18, 3-15, 9-21, 12-20
    sol_a = [(0, 12), (6, 18), (3, 15), (9, 21), (12, 20)]
    
    # Solution B: 4 edges
    # 0-12, 6-18, 12-20, 3-15
    sol_b = [(0, 12), (6, 18), (12, 20), (3, 15)]
    
    # Solution C: 4 edges different
    # 2-14, 8-20, 0-10, 14-22
    sol_c = [(2, 14), (8, 20), (0, 10), (14, 22)]
    
    # Solution D: 3 edges? Unlikely but try
    sol_d = [(0, 12), (6, 18), (12, 20)]
    
    # Solution E: 4 edges
    sol_e = [(1, 13), (7, 19), (13, 21), (4, 16)]
    
    # Solution F: 4 edges
    sol_f = [(0, 10), (10, 20), (5, 15), (15, 23)]
    
    test_solutions = [sol_d, sol_b, sol_c, sol_e, sol_f, sol_a]
    
    best_sol = None
    for sol in test_solutions:
        if time.time() > deadline:
            break
        if _valid_solution(n, base_edges, sol):
            if best_sol is None or len(sol) < len(best_sol):
                best_sol = sol
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    # Strategy 2: Greedy construction with local search
    # Start with no edges, greedily add edges that most reduce diameter
    added = []
    current_edges = list(base_edges)
    
    while time.time() < deadline:
        adj = _build_adj(n, current_edges)
        diam = _bfs_diameter(n, adj)
        
        if diam <= 4:
            break
        if diam == -1:
            break
        
        # Find the pair with maximum distance
        worst_u, worst_v = -1, -1
        max_dist_found = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_dist_found:
                            max_dist_found = dist[v]
                            worst_u = s
                            worst_v = v
        
        if worst_u == -1:
            break
        
        # Try all candidates, find the one that best reduces diameter
        best_candidate = None
        best_diam = max_dist_found + 1
        
        added_set = set((min(a,b), max(a,b)) for a,b in added)
        
        for a, b in candidates:
            if (min(a,b), max(a,b)) in added_set:
                continue
            # Temporarily add edge
            adj[a].add(b)
            adj[b].add(a)
            
            new_diam = _bfs_diameter(n, adj)
            
            # Remove edge
            adj[a].remove(b)
            adj[b].remove(a)
            
            if new_diam != -1 and new_diam < best_diam:
                best_diam = new_diam
                best_candidate = (a, b)
        
        if best_candidate is None:
            break
        
        added.append(best_candidate)
        current_edges.append(best_candidate)
    
    if _valid_solution(n, base_edges, added):
        if best_sol is None or len(added) < len(best_sol):
            best_sol = added
    
    # Strategy 3: Try to reduce the solution by removing edges
    if best_sol is not None:
        for i in range(len(best_sol) - 1, -1, -1):
            if time.time() > deadline:
                break
            reduced = best_sol[:i] + best_sol[i+1:]
            if _valid_solution(n, base_edges, reduced):
                best_sol = reduced
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    # Fallback: return empty (should not happen)
    return []

# EVOLVE-BLOCK-END
