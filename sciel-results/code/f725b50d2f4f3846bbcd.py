import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    m = 401
    
    def build_initial_graph():
        edges = set()
        # Cycle
        for i in range(n):
            edges.add((i, (i + 1) % n))
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        remaining = m - len(edges)
        attempts = 0
        while len(edges) < m and attempts < 100000:
            attempts += 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                continue
            
            edges.add((u, v))
            deg[u] += 1
            deg[v] += 1
        return edges

    def get_bitmasks(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def calc_book(u, v, adj):
        common = adj[u] & adj[v]
        return bin(common).count('1')

    def calc_all_books(edges, adj):
        books = {}
        for u, v in edges:
            books[(u, v)] = calc_book(u, v, adj)
        return books

    def find_worst_edge(books):
        worst = None
        max_val = -1
        for e, val in books.items():
            if val > max_val:
                max_val = val
                worst = e
        return worst, max_val

    def local_search(edges_set, adj, books, current_max, time_limit):
        """
        Optimized local search.
        """
        while time.time() < time_limit:
            worst_edge, worst_val = find_worst_edge(books)
            if worst_val <= 6:
                break
                
            u, v = worst_edge
            swapped = False
            
            # Try 50 random non-edges
            for _ in range(50):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) in edges_set:
                    continue
                
                # Tentative swap: remove (u,v), add (x,y)
                # 1. Update adj
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                # 2. Calculate new max book
                # We need to check all edges.
                # Edges are stored in edges_set.
                # We can just iterate over edges_set.
                # But we need to include (x,y) and exclude (u,v).
                # Actually, let's just create a temporary list of edges? No, iterate set.
                
                new_max = 0
                # Iterate over all edges in the graph (including the new one, excluding the removed one)
                # To do this efficiently without modifying the set:
                # We check every edge in edges_set.
                # If edge is (u,v), skip it (it's removed).
                # If edge is (x,y), include it (it's added).
                # Wait, (x,y) is NOT in edges_set yet.
                # So we iterate edges_set, skip (u,v), and also check (x,y) separately.
                
                for e in edges_set:
                    if e == (u, v):
                        continue
                    # Calculate book size for e
                    # Note: adj is already updated
                    val = calc_book(e[0], e[1], adj)
                    if val > new_max:
                        new_max = val
                
                # Check the new edge (x,y)
                val_xy = calc_book(x, y, adj)
                if val_xy > new_max:
                    new_max = val_xy
                
                if new_max < current_max:
                    # Accept
                    edges_set.remove((u, v))
                    edges_set.add((x, y))
                    
                    # Update books
                    # We only need to update books for edges that changed.
                    # But since we calculated new_max by iterating all, we can just rebuild books?
                    # Rebuilding books is O(m * n) which is ~400*40 = 16000 ops.
                    # Doing this every acceptance is fine if acceptances are rare.
                    # Or we can update incrementally.
                    # Let's update incrementally for speed.
                    # Affected edges: incident to u, v, x, y.
                    # But we already calculated new_max.
                    # We need to update the `books` dict.
                    
                    # Find affected edges
                    affected = set()
                    for w in [u, v, x, y]:
                        for z in range(n):
                            if (w, z) in edges_set:
                                if w < z:
                                    affected.add((w, z))
                                else:
                                    affected.add((z, w))
                    affected.add((x, y))
                    
                    # Update books for affected edges
                    for e in affected:
                        books[e] = calc_book(e[0], e[1], adj)
                        
                    # Remove (u,v) from books
                    if (u, v) in books:
                        del books[(u, v)]
                        
                    current_max = new_max
                    swapped = True
                    break
                else:
                    # Revert adj
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    
            if not swapped:
                # Stuck
                # Try to escape local minimum by random perturbation?
                # Or just break.
                # If we break, we restart.
                break

        return edges_set, current_max

    best_score = 999
    best_edges = None
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        edges = build_initial_graph()
        adj = get_bitmasks(edges)
        books = calc_all_books(edges, adj)
        current_max = max(books.values()) if books else 0
        
        if current_max < best_score:
            best_score = current_max
            best_edges = list(edges)
            
        if best_score <= 6:
            break
            
        # Local search
        # Allocate time based on remaining time
        remaining_time = deadline - 2 - time.time()
        time_per_restart = remaining_time / (num_restarts - i)
        if time_per_restart < 0.5:
            break
            
        edges, current_max = local_search(edges, adj, books, current_max, time.time() + time_per_restart)
        
        if current_max < best_score:
            best_score = current_max
            best_edges = list(edges)
            
        if best_score <= 6:
            break

    return [[u, v] for u, v in best_edges]
