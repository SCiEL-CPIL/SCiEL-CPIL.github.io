import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    # Start with K20,20 plus one edge (0,1)
    edge_set = set()
    for u in range(20):
        for v in range(20, 40):
            edge_set.add(frozenset((u, v)))
    edge_set.add(frozenset((0, 1)))
    
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for e in edge_set:
            a, b = tuple(e)
            adj[a].add(b)
            adj[b].add(a)
        return adj
    
    def compute_all_books(edge_set, adj):
        books = {}
        for e in edge_set:
            a, b = tuple(e)
            books[e] = len(adj[a] & adj[b])
        return books
    
    adj = build_adj(edge_set)
    books = compute_all_books(edge_set, adj)
    current_book = max(books.values())
    best_book = current_book
    best_edge_set = set(edge_set)
    
    random.seed(42)
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Bias towards edges with high book size
        # Filter edges with book size >= current_book - 2
        # If list is empty, fall back to random
        candidates = [e for e in edge_set if books[e] >= current_book - 2]
        if not candidates:
            candidates = list(edge_set)
        e_remove = random.choice(candidates)
        
        a, b = random.sample(range(n), 2)
        e_add = frozenset((a, b))
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
        
        u, v = tuple(e_remove)
        
        # Temporarily apply
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        adj[u].discard(v)
        adj[v].discard(u)
        adj[a].add(b)
        adj[b].add(a)
        
        affected = {e_add}
        for x in [u, v, a, b]:
            for y in adj[x]:
                e = frozenset((x, y))
                if e in edge_set:
                    affected.add(e)
        
        old_books = {e: books[e] for e in affected}
        
        for e in affected:
            x, y = tuple(e)
            books[e] = len(adj[x] & adj[y])
        
        new_global_max = max(books[e] for e in edge_set)
        
        delta = new_global_max - current_book
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_book = new_global_max
            if current_book < best_book:
                best_book = current_book
                best_edge_set = set(edge_set)
        else:
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            adj[a].discard(b)
            adj[b].discard(a)
            adj[u].add(v)
            adj[v].add(u)
            for e in affected:
                books[e] = old_books[e]
        
        T *= cooling
    
    return [list(e) for e in best_edge_set]
