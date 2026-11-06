import random
import numpy as np
import math

def construct():
    n = 40
    m = 401
    
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    # Helper to get list of edges from set
    def get_edge_list():
        return list(edges)

    # Helper to compute book sizes for all edges
    def compute_book_sizes():
        book_sizes = {}
        for u, v in edges:
            # Count common neighbors
            # adj[u] & adj[v] is a boolean array
            common = np.sum(adj[u] & adj[v])
            book_sizes[(u, v)] = common
        return book_sizes

    # Start with a random graph
    # Using a fixed seed for reproducibility in this context, 
    # but SA usually benefits from multiple restarts or good init.
    # Let's try a random init.
    random.seed(42)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    initial_edges = all_possible[:m]
    
    for u, v in initial_edges:
        adj[u, v] = True
        adj[v, u] = True
        edges.add((u, v))
        
    book_sizes = compute_book_sizes()
    current_max_book = max(book_sizes.values()) if book_sizes else 0
    
    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.99
    min_T = 0.1
    max_iterations = 20000
    
    for _ in range(max_iterations):
        if T < min_T:
            break
            
        # Pick a random edge to remove
        edge_list = list(edges)
        if not edge_list:
            break
        u, v = random.choice(edge_list)
        
        # Pick a random non-edge to add
        found = False
        while not found:
            x, y = random.sample(range(n), 2)
            if x > y:
                x, y = y, x
            if (x, y) not in edges:
                found = True
        
        # Calculate Delta E (change in max book size)
        # We need to update book sizes to know the new max.
        # Efficient update:
        # 1. Save old book sizes of affected edges
        # 2. Remove (u,v)
        # 3. Add (x,y)
        # 4. Update book sizes of affected edges
        # 5. Calculate new max
        # 6. If accepted, keep changes. Else revert.
        
        # Affected edges by removing (u,v):
        # Edges (u, z) where (v, z) in E
        # Edges (v, z) where (u, z) in E
        # And (u,v) itself is removed.
        
        # Affected edges by adding (x,y):
        # Edges (x, z) where (y, z) in E
        # Edges (y, z) where (x, z) in E
        # And (x,y) itself is added.
        
        # To simplify and ensure correctness, let's just recompute book sizes for all edges.
        # With n=40, m=401, recomputing takes ~401 * (40 ops) ~ 16k ops.
        # 20000 iterations * 16k ops = 320M ops. Might be slow in Python.
        
        # Let's try the incremental update.
        
        # 1. Identify affected edges
        affected_edges = set()
        # From removal of (u,v)
        for z in range(n):
            if adj[u, z] and adj[v, z] and (u, z) != (v, u) and (u, z) != (u, v):
                # Edge (min(u,z), max(u,z)) is affected
                a, b = (u, z) if u < z else (z, u)
                if (a, b) in edges:
                    affected_edges.add((a, b))
            if adj[v, z] and adj[u, z] and (v, z) != (u, v) and (v, z) != (u, v):
                a, b = (v, z) if v < z else (z, v)
                if (a, b) in edges:
                    affected_edges.add((a, b))
        
        # From addition of (x,y)
        for z in range(n):
            if adj[x, z] and adj[y, z]:
                a, b = (x, z) if x < z else (z, x)
                if (a, b) in edges:
                    affected_edges.add((a, b))
            if adj[y, z] and adj[x, z]:
                a, b = (y, z) if y < z else (z, y)
                if (a, b) in edges:
                    affected_edges.add((a, b))
        
        # Add the edges being swapped
        affected_edges.add((u, v))
        affected_edges.add((x, y))
        
        # Save old book sizes for these edges
        old_books = {}
        for e in affected_edges:
            if e in book_sizes:
                old_books[e] = book_sizes[e]
        
        # Perform swap
        adj[u, v] = False
        adj[v, u] = False
        edges.discard((u, v))
        
        adj[x, y] = True
        adj[y, x] = True
        edges.add((x, y))
        
        # Update book sizes for affected edges
        new_books = {}
        for e in affected_edges:
            if e in edges:
                a, b = e
                common = np.sum(adj[a] & adj[b])
                book_sizes[e] = common
                new_books[e] = common
            else:
                # Edge removed, remove from book_sizes
                if e in book_sizes:
                    del book_sizes[e]
        
        # Calculate new max book
        # We only changed books for affected_edges.
        # The new max is max(max of new_books, max of unchanged books)
        # To get max of unchanged books, we can keep a global max or re-scan.
        # Re-scanning 400 items is fast.
        new_max_book = max(book_sizes.values()) if book_sizes else 0
        
        delta = new_max_book - current_max_book
        
        # Simulated Annealing acceptance
        if delta <= 0:
            # Accept
            current_max_book = new_max_book
        else:
            # Accept with probability exp(-delta / T)
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_max_book = new_max_book
            else:
                # Revert
                adj[x, y] = False
                adj[y, x] = False
                edges.discard((x, y))
                
                adj[u, v] = True
                adj[v, u] = True
                edges.add((u, v))
                
                # Revert book sizes
                for e in affected_edges:
                    if e in old_books:
                        book_sizes[e] = old_books[e]
                    else:
                        # It was removed, so it shouldn't be in book_sizes now?
                        # Wait, if it was removed, it wasn't in edges, so not in book_sizes.
                        # If it was added, it wasn't in old_books.
                        # So we just restore what was there.
                        if e in book_sizes:
                            del book_sizes[e]
        
        T *= cooling_rate

    result = [list(e) for e in edges]
    return result
