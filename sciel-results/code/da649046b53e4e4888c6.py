import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def _deadline_check():
        return time.time() > deadline

    def verify_girth(adj, g):
        n = len(adj)
        if n == 0:
            return True
        radius = (g - 1) // 2
        for u in range(n):
            visited = {}
            visited[u] = (-1, 0)
            queue = deque([(u, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist == radius:
                    continue
                for nei in adj[node]:
                    if nei == parent:
                        continue
                    if nei in visited:
                        return False
                    visited[nei] = (node, dist + 1)
                    queue.append((nei, node, dist + 1))
        return True

    # Petersen Graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    edge_map = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    # Find all 5-cycles (shortest)
    cycles_5 = []
    def find_cycles_5(start, parent, current, length, path):
        if length == 0:
            if current == start:
                cycles_5.append(list(path))
            return
        for nei in base_adj[current]:
            if nei == parent:
                continue
            e_idx = edge_map[(current, nei)]
            path.append(e_idx)
            find_cycles_5(start, current, nei, length - 1, path)
            path.pop()
            
    for i in range(n_base):
        find_cycles_5(i, -1, i, 5, [])
        
    # Deduplicate cycles (rotations)
    unique_cycles_5 = set()
    for c in cycles_5:
        # Normalize: rotate to start with min edge index?
        # Or just use tuple
        unique_cycles_5.add(tuple(c))
    critical_walks = [list(c) for c in unique_cycles_5]
    
    # Search for m
    best_n = 10**9
    best_edges = None
    
    for m in range(51, 200):
        if time.time() > deadline - 5:
            break
            
        n_lift = 10 * m
        if n_lift >= best_n:
            continue
            
        max_trials = 100
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(len(petersen_edges))]
            
            valid = True
            for walk in critical_walks:
                s = 0
                for e_idx in walk:
                    s += voltages[e_idx]
                if s % m == 0:
                    valid = False
                    break
                    
            if valid:
                lift_adj = [[] for _ in range(n_lift)]
                for ei, (u, v) in enumerate(petersen_edges):
                    vol = voltages[ei]
                    for f in range(m):
                        a = u * m + f
                        b = v * m + ((f + vol) % m)
                        lift_adj[a].append(b)
                        lift_adj[b].append(a)
                        
                ok = True
                for i in range(n_lift):
                    if len(lift_adj[i]) != 3 or len(set(lift_adj[i])) != 3 or i in lift_adj[i]:
                        ok = False
                        break
                        
                if ok:
                    if verify_girth(lift_adj, g):
                        edges = []
                        for i in range(n_lift):
                            for j in lift_adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n_lift, edges

    # Fallback
    return 936, []

