import random
import time

def construct():
    deadline = time.time() + 130
    n = 40
    m = 401
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_books(adj):
        books = {}
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    books[(u, v)] = len(adj[u] & adj[v])
        return books
    
    def optimize(edges, adj, books, time_budget):
        current_max = max(books.values())
        best_max = current_max
        best_edges = list(edges)
        t0 = time.time()
        
        while time.time() - t0 < time_budget and time.time() < deadline:
            # Find worst edge
            worst_edge = max(books, key=lambda e: books[e])
            u, v = worst_edge
            
            # Try swaps
            found = False
            for _ in range(100):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                a, b = min(x, y), max(x, y)
                if (a, b) in adj[a]:
                    continue
                
                # Swap
                adj[u].discard(v)
                adj[v].discard(u)
                adj[a].add(b)
                adj[b].add(a)
                
                # Recompute affected
                affected = set()
                for w in adj[u]:
                    affected.add((min(u, w), max(u, w)))
                for w in adj[v]:
                    affected.add((min(v, w), max(v, w)))
                for w in adj[a]:
                    affected.add((min(a, w), max(a, w)))
                for w in adj[b]:
                    affected.add((min(b, w), max(b, w)))
                
                new_max = 0
                for e in affected:
                    x1, y1 = e
                    books[e] = len(adj[x1] & adj[y1])
                    if books[e] > new_max:
                        new_max = books[e]
                for e, bv in books.items():
                    if e not in affected and bv > new_max:
                        new_max = bv
                
                if new_max < current_max:
                    current_max = new_max
                    edges.remove((u, v))
                    edges.append((a, b))
                    if new_max < best_max:
                        best_max = new_max
                        best_edges = list(edges)
                    found = True
                    break
                elif new_max == current_max and random.random() < 0.1:
                    # Accept equal with small probability for exploration
                    current_max = new_max
                    edges.remove((u, v))
                    edges.append((a, b))
                    if new_max < best_max:
                        best_max = new_max
                        best_edges = list(edges)
                    found = True
                    break
                else:
                    # Revert
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[a].discard(b)
                    adj[b].discard(a)
            
            if not found:
                # Perturb: random swap
                u2, v2 = random.choice(list(edges))
                for _ in range(20):
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y:
                        continue
                    a, b = min(x, y), max(x, y)
                    if (a, b) in adj[a]:
                        continue
                    
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    adj[a].add(b)
                    adj[b].add(a)
                    
                    affected = set()
                    for w in adj[u2]:
                        affected.add((min(u2, w), max(u2, w)))
                    for w in adj[v2]:
                        affected.add((min(v2, w), max(v2, w)))
                    for w in adj[a]:
                        affected.add((min(a, w), max(a, w)))
                    for w in adj[b]:
                        affected.add((min(b, w), max(b, w)))
                    
                    new_max = 0
                    for e in affected:
                        x1, y1 = e
                        books[e] = len(adj[x1] & adj[y1])
                        if books[e] > new_max:
                            new_max = books[e]
                    for e, bv in books.items():
                        if e not in affected and bv > new_max:
                            new_max = bv
                    
                    if new_max <= current_max:
                        current_max = new_max
                        edges.remove((u2, v2))
                        edges.append((a, b))
                        if new_max < best_max:
                            best_max = new_max
                            best_edges = list(edges)
                        break
                    else:
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        adj[a].discard(b)
                        adj[b].discard(a)
        
        return best_edges, best_max
    
    best_overall = float('inf')
    best_edges_overall = None
    
    for start_idx in range(12):
        if time.time() > deadline - 15:
            break
        
        if start_idx == 0:
            edges = [(i, j) for i in range(20) for j in range(20, 40)]
            edges.append((0, 1))
        elif start_idx == 1:
            edges = [(i, j) for i in range(20) for j in range(20, 40)]
            for k in range(1, 10):
                edges.remove((0, 20 + k))
            for k in range(1, 11):
                edges.append((0, k))
        else:
            random.seed(start_idx * 7919 + 1)
            edges = random.sample(all_edges, m)
        
        edges = sorted(set((min(u, v), max(u, v)) for u, v in edges))
        if len(edges) != m:
            continue
        
        adj = build_adj(edges)
        books = compute_books(adj)
        
        found_edges, found_max = optimize(edges, adj, books, time_budget=8)
        
        if found_max < best_overall:
            best_overall = found_max
            best_edges_overall = found_edges
    
    if best_edges_overall is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_edges_overall]
