import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set_all = set(all_edges)
    
    # Use a set-based representation for fast adjacency checks
    # adj[u] = set of neighbors of u
    def make_adj_from_edges(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def book_size(adj, u, v):
        # Number of common neighbors of u and v
        return len(adj[u] & adj[v])
    
    def max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    def compute_all_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = len(adj[u] & adj[v])
        return books
    
    # Start with a random graph with m edges
    rng = random.Random(42)
    
    # Try to find a good starting point: balanced random
    best_edges = None
    best_max_book = 100
    
    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random graph
        edges = set(rng.sample(all_edges, m))
        adj = make_adj_from_edges(list(edges))
        mb = max_book(adj, list(edges))
        
        if mb < best_max_book:
            best_max_book = mb
            best_edges = list(edges)
        
        # Local search: hill climbing with random restarts
        current_edges = set(edges)
        current_adj = make_adj_from_edges(list(current_edges))
        current_mb = mb
        
        # Precompute books for current graph
        current_books = {}
        for u, v in current_edges:
            current_books[(u, v)] = len(current_adj[u] & current_adj[v])
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            # Pick a random edge to remove and a random non-edge to add
            edge_list = list(current_edges)
            if len(edge_list) < 2:
                break
            
            # Try a random swap
            for attempt in range(10):
                if time.time() > deadline - 3:
                    break
                
                # Remove a random edge
                e_remove = edge_list[random.randint(0, len(edge_list) - 1)]
                u1, v1 = e_remove
                
                # Add a random non-edge
                # Sample a random pair not in current_edges
                for _ in range(50):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) not in current_edges:
                        e_add = (u2, v2)
                        break
                else:
                    continue
                
                # Compute new book sizes for affected edges
                # Edges affected: e_remove, e_add, and edges sharing a vertex with either
                # For simplicity, just recompute max book for the new graph
                new_edges = set(current_edges)
                new_edges.remove(e_remove)
                new_edges.add(e_add)
                
                new_adj = make_adj_from_edges(list(new_edges))
                new_mb = max_book(new_adj, list(new_edges))
                
                if new_mb < current_mb:
                    current_edges = new_edges
                    current_adj = new_adj
                    current_mb = new_mb
                    no_improve = 0
                    if current_mb < best_max_book:
                        best_max_book = current_mb
                        best_edges = list(current_edges)
                    break
                else:
                    no_improve += 1
        
        if best_max_book <= 7:
            break
    
    if best_edges is not None and len(best_edges) == m:
        return [[u, v] for u, v in best_edges]
    
    # Fallback: random graph
    edges = set(random.Random(123).sample(all_edges, m))
    return [[u, v] for u, v in edges]
