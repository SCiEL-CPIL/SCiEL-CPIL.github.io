"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 40
    target_edges = 401
    
    # Start with K20,20 plus one edge
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((min(a,b), max(a,b)))
    edges_set.add((0, 1))
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_list = list(edges_set)
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    
    def compute_books(adj, edge_list):
        """Compute book size for each edge."""
        books = {}
        for u, v in edge_list:
            common = adj[u] & adj[v]
            books[(u, v)] = len(common)
        return books
    
    def max_book(books):
        return max(books.values()) if books else 0
    
    # Initial books
    books = compute_books(adj, edge_list)
    current_max = max_book(books)
    best_max = current_max
    best_edges = list(edge_list)
    
    # Local search: try swapping edges to reduce max book
    # Strategy: for edges with high book, try to remove them and add different edges
    
    random.seed(42)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Find edges with high book size
        high_book_edges = [(u, v) for (u, v), b in books.items() if b >= current_max - 1]
        
        if not high_book_edges:
            break
        
        # Pick a random high-book edge to modify
        u, v = random.choice(high_book_edges)
        
        # Try to remove edge (u,v) and add a new edge
        # The new edge should not create high books
        # Candidate: pick two vertices that are not adjacent and whose common neighbors are few
        
        # Try random swaps
        improved = False
        for _ in range(50):
            # Pick random non-edge
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or (x, y) in edges_set or (y, x) in edges_set:
                continue
            
            # Check if adding (x,y) would be beneficial
            # Remove (u,v), add (x,y)
            # Compute new book sizes for affected edges
            
            # Edges affected: (u,v) removed, (x,y) added
            # Also edges incident to u,v,x,y might change their book sizes
            
            # Quick check: book of (x,y) would be len(adj[x] & adj[y])
            new_book_xy = len(adj[x] & adj[y])
            
            if new_book_xy >= current_max:
                continue
            
            # Perform the swap temporarily
            adj[u].remove(v)
            adj[v].remove(u)
            edges_set.discard((u, v))
            
            adj[x].add(y)
            adj[y].add(x)
            edges_set.add((min(x,y), max(x,y)))
            
            # Recompute books for affected edges
            # Affected: all edges incident to u, v, x, y
            affected = set()
            for w in adj[u]:
                affected.add((min(u,w), max(u,w)))
            for w in adj[v]:
                affected.add((min(v,w), max(v,w)))
            for w in adj[x]:
                affected.add((min(x,w), max(x,w)))
            for w in adj[y]:
                affected.add((min(y,w), max(y,w)))
            
            new_books = {}
            for (a, b) in affected:
                if (a, b) in edges_set:
                    new_books[(a, b)] = len(adj[a] & adj[b])
            
            # Update books
            for (a, b), bsize in new_books.items():
                books[(a, b)] = bsize
            
            # Remove old book for (u,v)
            if (u, v) in books:
                del books[(u, v)]
            
            new_max = max(books.values()) if books else 0
            
            if new_max < current_max:
                current_max = new_max
                best_max = min(best_max, current_max)
                best_edges = list(edges_set)
                improved = True
                break
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add((u, v))
                
                adj[x].remove(y)
                adj[y].remove(x)
                edges_set.discard((min(x,y), max(x,y)))
                
                # Revert books
                for (a, b) in affected:
                    if (a, b) in edges_set:
                        books[(a, b)] = len(adj[a] & adj[b])
                    elif (a, b) in books:
                        del books[(a, b)]
                books[(u, v)] = len(adj[u] & adj[v])
        
        if not improved and iteration % 100 == 0:
            # Try a more aggressive restart: perturb the graph
            # Remove 5 random edges and add 5 new ones
            for _ in range(5):
                if len(edges_set) <= target_edges:
                    break
                rem_u, rem_v = random.choice(list(edges_set))
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                edges_set.discard((rem_u, rem_v))
                if (rem_u, rem_v) in books:
                    del books[(rem_u, rem_v)]
            
            # Add random edges
            attempts = 0
            while len(edges_set) < target_edges and attempts < 1000:
                attempts += 1
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y or (x, y) in edges_set or (y, x) in edges_set:
                    continue
                edges_set.add((min(x,y), max(x,y)))
                adj[x].add(y)
                adj[y].add(x)
            
            # Recompute all books
            books = {}
            for (a, b) in edges_set:
                books[(a, b)] = len(adj[a] & adj[b])
            current_max = max(books.values()) if books else 0
            best_max = min(best_max, current_max)
            if current_max < best_max:
                best_edges = list(edges_set)
    
    # Return the best found
    if best_edges:
        return [[u, v] for u, v in best_edges]
    else:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
# EVOLVE-BLOCK-END
