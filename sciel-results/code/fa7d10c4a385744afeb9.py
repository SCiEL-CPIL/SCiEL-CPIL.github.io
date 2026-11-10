"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 75
    n = P['n']
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_diam2(adj):
        for src in range(n):
            dist = [-1]*n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node]+1
                        queue.append(nb)
            if -1 in dist:
                return False
            for d in dist:
                if d > 2:
                    return False
        return True
    
    def is_critical(edges, u, v):
        adj = build_adj(edges)
        adj[u].discard(v)
        adj[v].discard(u)
        for src in range(n):
            dist = [-1]*n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node]+1
                        queue.append(nb)
            if -1 in dist:
                return True
            for d in dist:
                if d > 2:
                    return True
        return False
    
    def all_critical(edges):
        for u,v in edges:
            if not is_critical(edges, u, v):
                return False
        return True
    
    def greedy_add(edges):
        edge_set = set()
        for u,v in edges:
            edge_set.add((min(u,v), max(u,v)))
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i,j))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = sorted(all_possible - edge_set)
            random.shuffle(candidates)
            for e in candidates:
                if time.time() > deadline:
                    break
                test = list(edges) + [e]
                if all_critical(test):
                    edges.append(e)
                    edge_set.add(e)
                    improved = True
                    break
        return edges
    
    best = []
    
    # Start with star
    star = [(0, v) for v in range(1, n)]
    result = greedy_add(list(star))
    if len(result) > len(best):
        best = result
    
    # Try random initial graphs
    for attempt in range(200):
        if time.time() > deadline:
            break
        random.seed(attempt * 7919 + 13)
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i,j))
        edges = []
        for e in all_possible:
            if random.random() < 0.2:
                edges.append(e)
        adj = build_adj(edges)
        if not check_diam2(adj):
            for e in sorted(all_possible - set(edges)):
                if time.time() > deadline:
                    break
                test = edges + [e]
                adj_test = build_adj(test)
                if check_diam2(adj_test):
                    edges.append(e)
        adj = build_adj(edges)
        if not check_diam2(adj):
            continue
        if len(edges) == n*(n-1)//2:
            continue
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(edges)-1, -1, -1):
                u,v = edges[i]
                if not is_critical(edges, u, v):
                    edges.pop(i)
                    changed = True
                    break
        adj = build_adj(edges)
        if not check_diam2(adj):
            continue
        if len(edges) == n*(n-1)//2:
            continue
        result = greedy_add(list(edges))
        if len(result) > len(best):
            best = result
    
    if not best:
        best = star
    return [[int(u), int(v)] for u,v in best]
# EVOLVE-BLOCK-END
