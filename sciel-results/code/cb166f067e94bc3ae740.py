import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    seed_edges = set()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                seed_edges.add((5 * k + a, 5 * k + b))
    
    best_edges = seed_edges.copy()
    best_count = len(best_edges)
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_path_len_5(u, v, adj):
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist == 5:
                if node == v:
                    return True
                continue
            for nei in adj[node]:
                if nei not in visited:
                    visited[nei] = dist + 1
                    if dist + 1 <= 5:
                        queue.append((nei, dist + 1))
        return False

    def greedy_add(initial_edges, candidate_edges):
        current_edges = set(initial_edges)
        adj = build_adj(current_edges)
        
        for e in candidate_edges:
            if e in current_edges:
                continue
            u, v = e
            if not has_path_len_5(u, v, adj):
                current_edges.add(e)
                adj[u].add(v)
                adj[v].add(u)
                
        return current_edges

    while time.time() < deadline:
        random.shuffle(all_edges)
        
        g1 = greedy_add(set(), all_edges)
        if len(g1) > best_count:
            best_count = len(g1)
            best_edges = g1
            
        g2 = greedy_add(seed_edges, all_edges)
        if len(g2) > best_count:
            best_count = len(g2)
            best_edges = g2
            
        if time.time() > deadline - 5.0:
            break

    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
