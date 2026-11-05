import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def build_adj(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book_size(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def compute_books(edge_set, adj):
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = book_size(u, v, adj)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def random_regular_init():
        """
        Initialize with a random circulant graph (which is 20-regular)
        and add 1 random edge.
        """
        edge_set = set()
        # Pick 10 random distinct shifts from 1 to 19
        shifts = random.sample(range(1, 20), 10)
        for shift in shifts:
            for i in range(n):
                j = (i + shift) % n
                if i < j:
                    edge_set.add((i, j))
                else:
                    edge_set.add((j, i))
        
        # We have 400 edges. Add 1 more.
        available = [e for e in all_edges if e not in edge_set]
        if available:
            edge_set.add(random.choice(available))
        
        return edge_set

    def simulated_annealing(edge_set, adj, books, current_score, max_steps=20000):
        """
        Simulated Annealing to minimize the max book size.
        """
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        best_score = current_score
        best_books = books.copy()
        best_edge_set = set(edge_set)
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            if T < min_T:
                break
                
            # Pick a random edge to remove
            u, v = random.choice(list(edge_set))
            
            # Pick a random non-edge to add
            # Sample until we find one
            added = False
            for _ in range(50):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) not in edge_set:
                    # Apply move
                    edge_set.remove((u, v))
                    edge_set.add((ru, rv))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    
                    # Calculate new books for affected edges
                    affected_vertices = {u, v, ru, rv}
                    new_books = books.copy()
                    
                    if (u, v) in new_books:
                        del new_books[(u, v)]
                    
                    new_books[(ru, rv)] = book_size(ru, rv, adj)
                    
                    edges_to_check = set()
                    for vert in affected_vertices:
                        neighbors = [i for i in range(n) if (adj[vert] >> i) & 1]
                        for nb in neighbors:
                            if vert < nb:
                                edges_to_check.add((vert, nb))
                            else:
                                edges_to_check.add((nb, vert))
                    
                    for e in edges_to_check:
                        if e in new_books:
                            new_books[e] = book_size(e[0], e[1], adj)
                    
                    new_score = max_book(new_books)
                    
                    # Acceptance criterion
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        current_score = new_score
                        books = new_books
                        if new_score < best_score:
                            best_score = new_score
                            best_books = new_books.copy()
                            best_edge_set = set(edge_set)
                    else:
                        # Revert
                        edge_set.remove((ru, rv))
                        edge_set.add((u, v))
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                    added = True
                    break
            
            if not added:
                continue
                
            T *= cooling
            
        return best_edge_set, best_score, best_books

    best_score = 999
    best_edge_set = None
    best_books = None

    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        edge_set = random_regular_init()
        adj = build_adj(edge_set)
        books = compute_books(edge_set, adj)
        current_score = max_book(books)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_books = books.copy()

        if current_score <= 7:
            break

        # Run Simulated Annealing
        edge_set, current_score, books = simulated_annealing(edge_set, adj, books, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_books = books.copy()

        if current_score <= 7:
            break

    if best_edge_set is None:
        best_edge_set = random_regular_init()

    return [[u, v] for (u, v) in best_edge_set]
