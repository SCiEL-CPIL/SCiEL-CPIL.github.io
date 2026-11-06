import random
import time
import numpy as np

def _compute_max_book(edges, n=40):
    """Compute the largest book size for a given set of edges."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v) if u < v else (v, u))
    
    max_book = 0
    for u, v in edge_set:
        common = len(adj[u].intersection(adj[v]))
        if common > max_book:
            max_book = common
    return max_book

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Start with a balanced bipartite graph K_{20,20} which has 400 edges.
    # Then add 1 edge within one partition.
    # In K_{20,20}, the book size for any cross edge is 0 (no triangles).
    # If we add an intra-partition edge (u,v) in partition A:
    # - The edge (u,v) itself is in no triangles initially? No, wait.
    # - Triangles containing (u,v): A triangle (u,v,w) requires edges (u,w) and (v,w).
    #   Since u,v are in A, and w must be in B (because there are no other edges in A),
    #   (u,w) and (v,w) are cross edges, which exist in K_{20,20}.
    #   So every w in B forms a triangle with (u,v).
    #   Thus, the book size of the new edge (u,v) is |B| = 20.
    #   This is too high. We want to minimize the max book.
    
    # Let's rethink.
    # The lower bound is 40/6 ≈ 6.67, so min integer is 7.
    # K_{20,20} + 1 edge gives max book 20. This is bad.
    
    # We need a denser structure that allows triangles to be shared more evenly or simply fewer.
    # Actually, the problem is minimizing the *maximum* book.
    # To get a small max book, we need to avoid edges that are in many triangles.
    # This suggests a graph that is "triangle-sparse" locally, or where triangles are distributed.
    
    # Consider the complement.
    # Or consider a random graph G(n, p).
    # Expected triangles containing an edge: p^2 * (n-2).
    # We want this to be small.
    # But we have fixed m=401 edges.
    # p = 2m / (n(n-1)) = 802 / (40*39) = 802 / 1560 ≈ 0.514.
    # Expected book size ≈ 0.514^2 * 38 ≈ 0.264 * 38 ≈ 10.
    # So a random graph likely has max book around 10-12.
    
    # Can we do better?
    # The Edwards-Khadzhiivanov-Nikiforov bound says max book >= n/6 ≈ 6.67.
    # So 7 is the theoretical minimum.
    # Is 7 achievable?
    
    # Let's try a different construction.
    # Start with a random graph and use local search to reduce the max book.
    # Key insight from the failed attempt: The local search was too slow because it recomputed everything.
    # We need an efficient way to update the book sizes.
    
    # Efficient Local Search:
    # 1. Maintain adjacency sets.
    # 2. Maintain book_size[u, v] for all edges (u,v).
    # 3. When adding/removing an edge, update book sizes for affected edges.
    
    # Let's implement a faster local search.
    
    # Initialize with a random graph
    random.seed(42)
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Generate 401 random edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    
    for u, v in all_possible[:401]:
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute initial book sizes
    book_size = {}
    for u, v in edges:
        book_size[(u, v)] = len(adj[u].intersection(adj[v]))
    
    max_book = max(book_size.values())
    best_edges = set(edges)
    best_max_book = max_book
    
    # Local search
    # Move: Remove edge (u,v), add edge (x,y).
    # We want to reduce the max book.
    
    # To speed up, we can precompute candidates.
    # But for now, let's try random moves with efficient updates.
    
    def update_book_on_add(u, v):
        """Add edge (u,v) and update book sizes for edges involving u or v."""
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
        # New edge (u,v) has book size = |adj[u] ∩ adj[v]|
        book_size[(u, v)] = len(adj[u].intersection(adj[v]))
        
        # Update book sizes for edges (u, w) and (v, w)
        for w in adj[u]:
            if w != v:
                # Book size of (u, w) increases if w is connected to v
                # Actually, book size of (u, w) is |adj[u] ∩ adj[w]|
                # Since we added (u,v), adj[u] changed.
                # The new common neighbors are the old ones plus v if v is in adj[w].
                if v in adj[w]:
                    book_size[(u, w)] = book_size.get((u, w), 0) + 1
                else:
                    # No change? Wait.
                    # adj[u] gained v.
                    # Common neighbors of u and w: old_common ∩ adj[w] ∪ {v if v in adj[w]}
                    # So if v in adj[w], size increases by 1.
                    # If v not in adj[w], size stays same.
                    pass
        
        for w in adj[v]:
            if w != u:
                if u in adj[w]:
                    book_size[(v, w)] = book_size.get((v, w), 0) + 1
                else:
                    pass
    
    def update_book_on_remove(u, v):
        """Remove edge (u,v) and update book sizes for edges involving u or v."""
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        
        del book_size[(u, v)]
        
        for w in adj[u]:
            if w != v:
                if v in adj[w]:
                    book_size[(u, w)] = book_size.get((u, w), 1) - 1
        
        for w in adj[v]:
            if w != u:
                if u in adj[w]:
                    book_size[(v, w)] = book_size.get((v, w), 1) - 1
    
    # We need to be careful with the logic above.
    # Let's simplify: Just recompute book sizes for neighbors of u and v.
    # This is O(deg(u) + deg(v)) per move, which is fast.
    
    def recompute_books_for(u, v):
        """Recompute book sizes for all edges incident to u or v."""
        # Edges incident to u
        for w in adj[u]:
            key = (u, w) if u < w else (w, u)
            book_size[key] = len(adj[u].intersection(adj[w]))
        # Edges incident to v
        for w in adj[v]:
            key = (v, w) if v < w else (w, v)
            book_size[key] = len(adj[v].intersection(adj[w]))
    
    # Let's restart with a cleaner implementation.
    # The previous logic was getting complex.
    
    # Re-initialize
    edges = set()
    adj = [set() for _ in range(n)]
    for u, v in all_possible[:401]:
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    book_size = {}
    for u, v in edges:
        book_size[(u, v)] = len(adj[u].intersection(adj[v]))
    
    max_book = max(book_size.values())
    best_edges = set(edges)
    best_max_book = max_book
    
    # Local Search Loop
    while time.time() < deadline:
        # Pick a random edge to remove
        u_rem, v_rem = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # Sample until we find a non-edge
        added = False
        for _ in range(50):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edges:
                # Perform move
                # 1. Remove (u_rem, v_rem)
                edges.remove((u_rem, v_rem))
                adj[u_rem].remove(v_rem)
                adj[v_rem].remove(u_rem)
                del book_size[(u_rem, v_rem)]
                
                # 2. Add (x, y)
                edges.add((x, y))
                adj[x].add(y)
                adj[y].add(x)
                
                # 3. Update book sizes for neighbors of u_rem, v_rem, x, y
                # We need to recompute for all edges incident to these 4 vertices
                # To be safe and simple, we recompute for all edges incident to any of the 4 vertices.
                # Actually, changing an edge (u,v) affects the book size of edges (u,w) and (v,w).
                # So removing (u_rem, v_rem) affects edges incident to u_rem and v_rem.
                # Adding (x, y) affects edges incident to x and y.
                
                # Let's just recompute book sizes for all edges incident to u_rem, v_rem, x, y.
                # This is a small set.
                affected_vertices = {u_rem, v_rem, x, y}
                affected_edges = set()
                for v in affected_vertices:
                    for w in adj[v]:
                        key = (v, w) if v < w else (w, v)
                        affected_edges.add(key)
                
                # Save old values
                old_values = {key: book_size.get(key, 0) for key in affected_edges}
                
                # Recompute
                for key in affected_edges:
                    a, b = key
                    book_size[key] = len(adj[a].intersection(adj[b]))
                
                # Check if max book decreased
                # We need to know the new max book.
                # We can keep a counter of book sizes or just compute max over affected_edges and global max.
                # To be efficient, we can maintain a max-heap or just check if any affected edge exceeds current max.
                # But we also need to know if the global max was removed.
                
                # Simple approach: Compute max over all edges.
                # This is O(E) = 401.
                # 401 * 50 (sampling) * (moves/sec)
                # If we do 1000 moves/sec, that's 401k operations/sec.
                # 110 seconds -> 44 million operations.
                # This is feasible in Python? Maybe slow.
                
                # Let's optimize:
                # Keep a variable `current_max_book`.
                # If the removed edge was the unique max, we need to find the new max.
                # If the added edge is less than current max, and removed edge was not max, max stays same.
                
                # Let's just compute max over all edges for now, it's only 401 edges.
                new_max_book = max(book_size.values()) if book_size else 0
                
                if new_max_book < best_max_book:
                    best_max_book = new_max_book
                    best_edges = set(edges)
                elif new_max_book == max_book:
                    # Accept with some probability? Or always accept to explore?
                    # Hill climbing: accept if better or equal?
                    # If equal, we might get stuck. Let's accept if better, or with small prob if equal.
                    if random.random() < 0.1:
                        max_book = new_max_book
                    else:
                        # Revert
                        edges.remove((x, y))
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        edges.add((u_rem, v_rem))
                        adj[u_rem].add(v_rem)
                        adj[v_rem].add(u_rem)
                        
                        # Restore book sizes
                        for key, val in old_values.items():
                            book_size[key] = val
                else:
                    # Revert
                    edges.remove((x, y))
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    edges.add((u_rem, v_rem))
                    adj[u_rem].add(v_rem)
                    adj[v_rem].add(u_rem)
                    
                    # Restore book sizes
                    for key, val in old_values.items():
                        book_size[key] = val
                
                max_book = new_max_book if new_max_book <= best_max_book else max_book
                # Actually, we should track current state's max book properly.
                # Let's just recompute current_max_book after each move.
                
                added = True
                break
        
        if not added:
            continue
    
    # Convert best_edges back to list of lists
    final_edges = [[u, v] for u, v in best_edges]
    
    # Ensure exactly 401 edges
    if len(final_edges) != 401:
        # Fallback
        return [[u, v] for u, v in edges]
        
    return final_edges
