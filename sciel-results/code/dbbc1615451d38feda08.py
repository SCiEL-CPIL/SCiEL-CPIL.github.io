import time
import random

DEADLINE = 84.0

def _is_valid(adj, deg, n):
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def construct():
    n = 15
    t0 = time.time()
    
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
    
    def edge_count():
        return sum(len(s) for s in adj) // 2
    
    # Start with star
    for v in range(1, n):
        add_edge(0, v)
    
    # Greedy fill
    for _ in range(10):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        add_edge(u, v)
                        if _is_valid(adj, deg, n):
                            changed = True
                        else:
                            remove_edge(u, v)
    
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    best_count = edge_count()
    
    # Simulated annealing
    T = 3.0
    while time.time() - t0 < DEADLINE:
        r = random.random()
        if r < 0.45:
            u, v = random.sample(range(n), 2)
            if v not in adj[u]:
                add_edge(u, v)
                if _is_valid(adj, deg, n):
                    c = edge_count()
                    if c > best_count:
                        best_adj = [set(s) for s in adj]
                        best_deg = list(deg)
                        best_count = c
                else:
                    remove_edge(u, v)
        elif r < 0.65:
            edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if edges:
                u, v = random.choice(edges)
                remove_edge(u, v)
        else:
            edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if edges:
                u1, v1 = random.choice(edges)
                u2, v2 = random.sample(range(n), 2)
                if u2 > v2:
                    u2, v2 = v2, u2
                if v2 not in adj[u2]:
                    remove_edge(u1, v1)
                    add_edge(u2, v2)
                    if _is_valid(adj, deg, n):
                        c = edge_count()
                        if c > best_count:
                            best_adj = [set(s) for s in adj]
                            best_deg = list(deg)
                            best_count = c
                    else:
                        remove_edge(u2, v2)
                        add_edge(u1, v1)
        T *= 0.999
        if T < 0.01:
            T = 0.01
    
    # Final greedy polish
    adj2 = [set(s) for s in best_adj]
    deg2 = list(best_deg)
    for _ in range(5):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj2[u]:
                        adj2[u].add(v)
                        adj2[v].add(u)
                        deg2[u] += 1
                        deg2[v] += 1
                        if _is_valid(adj2, deg2, n):
                            changed = True
                        else:
                            adj2[u].discard(v)
                            adj2[v].discard(u)
                            deg2[u] -= 1
                            deg2[v] -= 1
    
    result = []
    for u in range(n):
        for v in adj2[u]:
            if u < v:
                result.append([int(u), int(v)])
    return result
