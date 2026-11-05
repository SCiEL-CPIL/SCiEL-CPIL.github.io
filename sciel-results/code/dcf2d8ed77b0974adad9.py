import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Adjacency as list of bitmasks for fast common-neighbor computation
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
        
    def get_book(u, v):
        # Common neighbors of u and v
        return (adj[u] & adj[v]).bit_count()
        
    def compute_max_book(edges):
        max_b = 0
        for u, v in edges:
            b = get_book(u, v)
            if b > max_b:
                max_b = b
        return max_b

    def get_all_edges():
        edges = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.add((u, v))
                mask -= lsb
        return edges

    def compute_books_dict(edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = get_book(u, v)
        return books

    # Structural initialization: K20,20 + two K10 cliques
    # Left side: 0..19, split into L1 (0..9) and L2 (10..19)
    # Right side: 20..39, split into R1 (20..29) and R2 (30..39)
    # Base: K20,20 (all edges between left and right) = 400 edges
    # Add: edges within L1 (K10) and within R1 (K10)
    # But we need 401 edges. K20,20 has 400 edges.
    # So we start with K20,20 + 1 extra edge.
    # The directive suggests: "partitioning each side into two groups of 10 and adding all edges within each group (creating two disjoint K10s)"
    # That would be 400 + 45 + 45 = 490 edges. Too many.
    # We need to remove 490 - 401 = 89 edges.
    # Alternatively, start with K20,20 (400 edges) + 1 edge from one of the cliques.
    # Let's try the directive's structure but remove edges to get to 401.
    # Actually, let's just start with K20,20 + 1 random edge, which is a valid 401-edge graph.
    # Then use local search to optimize.
    
    def create_initial_graph(seed_val):
        random.seed(seed_val)
        edges = set()
        
        # K20,20
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        
        # Add 1 extra edge to make 401
        # Pick a random non-edge. Non-edges are within left or within right.
        # Let's pick an edge within left side to break symmetry slightly
        # Or just pick a random pair from all non-edges
        all_non_edges = []
        for u in range(20):
            for v in range(u+1, 20):
                all_non_edges.append((u, v))
        for u in range(20, 40):
            for v in range(u+1, 40):
                all_non_edges.append((u, v))
        
        if all_non_edges:
            extra = random.choice(all_non_edges)
            edges.add(extra)
        
        return edges

    def optimize(edges_set, time_limit):
        # Initialize adj
        for i in range(n):
            adj[i] = 0
        for u, v in edges_set:
            add_edge(u, v)
            
        current_edges = set(edges_set)
        books = compute_books_dict(current_edges)
        current_max = max(books.values()) if books else 0
        
        best_edges = set(current_edges)
        best_max = current_max
        best_books = dict(books)
        
        # Simulated Annealing with bitmask acceleration
        # Key insight: when we remove edge (u,v) and add edge (a,b),
        # only books of edges incident to u, v, a, b can change.
        
        temp = 5.0
        cooling = 0.9995
        
        while time.time() < time_limit:
            # Find edge with max book
            max_b = -1
            worst_edge = None
            for e, b in books.items():
                if b > max_b:
                    max_b = b
                    worst_edge = e
            
            if worst_edge is None:
                break
                
            u, v = worst_edge
            
            # Generate a candidate replacement edge
            found_new_edge = False
            for _ in range(50):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                new_edge = (a, b)
                if new_edge not in current_edges and new_edge != worst_edge:
                    found_new_edge = True
                    break
            
            if not found_new_edge:
                break
            
            # Perform swap: remove (u,v), add (a,b)
            remove_edge(u, v)
            add_edge(a, b)
            
            # Update books incrementally
            # Only edges incident to u, v, a, b need updating
            # Edges incident to u: (u, x) for x in N(u)
            # Edges incident to v: (v, x) for x in N(v)
            # Edges incident to a: (a, x) for x in N(a)
            # Edges incident to b: (b, x) for x in N(b)
            
            # Collect all edges that might have changed
            affected_edges = set()
            
            # Edges incident to u
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                x = lsb.bit_length() - 1
                if u < x:
                    affected_edges.add((u, x))
                else:
                    affected_edges.add((x, u))
                mask_u -= lsb
            
            # Edges incident to v
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                x = lsb.bit_length() - 1
                if v < x:
                    affected_edges.add((v, x))
                else:
                    affected_edges.add((x, v))
                mask_v -= lsb
            
            # Edges incident to a
            mask_a = adj[a]
            while mask_a:
                lsb = mask_a & -mask_a
                x = lsb.bit_length() - 1
                if a < x:
                    affected_edges.add((a, x))
                else:
                    affected_edges.add((x, a))
                mask_a -= lsb
            
            # Edges incident to b
            mask_b = adj[b]
            while mask_b:
                lsb = mask_b & -mask_b
                x = lsb.bit_length() - 1
                if b < x:
                    affected_edges.add((b, x))
                else:
                    affected_edges.add((x, b))
                mask_b -= lsb
            
            # Update books for affected edges
            for e in affected_edges:
                if e in current_edges:
                    books[e] = get_book(*e)
                elif e in books:
                    del books[e]
            
            # Add the new edge's book
            books[(a, b)] = get_book(a, b)
            
            # Remove the old edge's book if not already removed
            if worst_edge in books:
                del books[worst_edge]
            
            # Update current_edges
            current_edges.remove(worst_edge)
            current_edges.add(new_edge)
            
            new_max = max(books.values()) if books else 0
            
            delta = new_max - current_max
            
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(current_edges)
                    best_books = dict(books)
            else:
                # Revert
                remove_edge(a, b)
                add_edge(u, v)
                current_edges.remove(new_edge)
                current_edges.add(worst_edge)
                
                # Revert books
                # Re-add old edge book
                books[worst_edge] = get_book(u, v)
                # Remove new edge book
                if (a, b) in books:
                    del books[(a, b)]
                
                # Re-update affected edges
                for e in affected_edges:
                    if e in current_edges:
                        books[e] = get_book(*e)
                    elif e in books:
                        del books[e]
                
                # Restore current_max
                current_max = max(books.values()) if books else 0
                
            temp *= cooling
            
        return best_edges, best_max

    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts
    seed = 42
    while time.time() < deadline - 10.0:
        initial_edges = create_initial_graph(seed)
        optimized_edges, score = optimize(initial_edges, deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_edges = optimized_edges
            if best_score <= 6:
                break
        
        seed += 1
    
    if best_edges is None:
        # Fallback: K20,20 + 1 edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_edges]
