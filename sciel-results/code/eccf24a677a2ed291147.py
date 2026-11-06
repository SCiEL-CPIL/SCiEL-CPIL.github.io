import random
import math

def construct():
    n = 40
    target_edges = 401
    
    def compute_books(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        books = {}
        for u, v in edge_set:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    def get_max_book(books):
        if not books:
            return 0
        return max(books.values())

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    # Initialize with Turan graph T_40(2) plus 1 edge
    # Partitions A and B of size 20
    A = list(range(20))
    B = list(range(20, 40))
    
    current_edges = set()
    for u in A:
        for v in B:
            current_edges.add((u, v))
    
    # Add one edge within A to reach 401 edges
    # This creates a book of size 20 for this edge
    current_edges.add((0, 1))
    
    current_books = compute_books(current_edges)
    current_cost = get_sum_sq_books(current_books)
    best_edges = set(current_edges)
    best_cost = current_cost
    best_max_book = get_max_book(current_books)
    
    rng = random.Random(42)
    
    # Local Search / Simulated Annealing hybrid
    # Focus on reducing the maximum book size
    
    T = 50.0
    cooling = 0.9995
    min_T = 0.01
    
    for i in range(100000):
        if T < min_T:
            break
            
        # Pick an edge to remove, preferably one with high book size
        # Sort edges by book size descending
        edge_list = list(current_edges)
        # To avoid sorting every time, we can just pick random but weight by book size?
        # Or just pick random, as SA explores.
        # But for speed, let's pick a random edge.
        rem_u, rem_v = edge_list[rng.randrange(len(edge_list))]
        
        # Pick a random edge to add
        add_u = rng.randrange(n)
        add_v = rng.randrange(n)
        if add_u == add_v:
            continue
        add_u, add_v = min(add_u, add_v), max(add_u, add_v)
        if (add_u, add_v) in current_edges:
            continue
            
        new_edges = current_edges.copy()
        new_edges.discard((rem_u, rem_v))
        new_edges.add((add_u, add_v))
        
        new_books = compute_books(new_edges)
        new_cost = get_sum_sq_books(new_books)
        new_max_book = get_max_book(new_books)
        
        delta = new_cost - current_cost
        
        # Acceptance probability
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_edges = new_edges
            current_books = new_books
            current_cost = new_cost
            
            if new_max_book < best_max_book or (new_max_book == best_max_book and new_cost < best_cost):
                best_max_book = new_max_book
                best_cost = new_cost
                best_edges = new_edges
                
                # If we reached the theoretical lower bound, we can stop or continue to polish
                if best_max_book <= 7:
                    # Continue for a bit to ensure we have the best cost among 7s, 
                    # but 7 is the min possible, so we just need to be stable.
                    # We can break early if we want to save time, but let's run a bit more.
                    pass
        
        T *= cooling

    return [[u, v] for u, v in best_edges]
