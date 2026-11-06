import time
import random
import sys

def construct():
    """
    Construct a graph with 40 vertices and 401 edges that minimizes the largest book size.
    The largest book size is the maximum over all edges (u,v) of the number of common neighbors of u and v.
    """
    n = 40
    m = 401
    
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s wall-clock limit
    
    def compute_books(adj, edges):
        """Compute the book size for each edge."""
        book = {}
        for u, v in edges:
            if u > v:
                u, v = v, u
            common = adj[u] & adj[v]
            book[(u, v)] = len(common)
        return book
    
    def max_book(book):
        """Get the maximum book size."""
        if not book:
            return 0
        return max(book.values())
    
    def generate_random_graph(n, m):
        """Generate a random graph with n vertices and m edges."""
        vertices = list(range(n))
        edges = set()
        
        # Start with a random spanning tree to ensure connectivity (optional, but helpful)
        # Actually, for this problem, we don't strictly need connectivity, but it helps with structure.
        # Let's just pick random edges.
        
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_possible)
        
        for e in all_possible[:m]:
            edges.add(e)
        
        # Ensure exactly m edges
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        
        # Convert to list
        return list(edges)
    
    def local_search(initial_edges, max_iter, time_limit_check):
        """
        Perform local search by swapping edges to minimize the maximum book size.
        """
        edges_set = set(initial_edges)
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute initial books
        book = compute_books(adj, edges_set)
        current_max = max_book(book)
        best_max = current_max
        best_edges = list(edges_set)
        
        # We use a simple approach: try random edge swaps and accept if it improves or keeps the same
        # To speed up, we can maintain the book dictionary and update it incrementally.
        # However, for simplicity and robustness within the time limit, we will recompute
        # but optimize the inner loop.
        
        # Incremental update approach:
        # When removing edge (u,v):
        #   For each w in adj[u] & adj[v], book[(u,w)] decreases by 1, book[(v,w)] decreases by 1.
        #   Also, for each w in adj[u] (excluding v), if w is connected to v? No, just update common neighbors.
        #   Actually, the book size of an edge (x,y) is |N(x) ∩ N(y)|.
        #   Removing (u,v) affects:
        #   1. The edge (u,v) itself is removed.
        #   2. For any w, if (u,w) is an edge, N(u) loses v, so if v is in N(w), then book[(u,w)] decreases by 1.
        #   3. Similarly for (v,w).
        #   4. For any w, if (w, x) and (w, y) ... no, the edges are between existing vertices.
        
        # Let's stick to a hybrid: 
        # 1. Pick a random edge to remove.
        # 2. Pick a random non-edge to add.
        # 3. Calculate the change in the maximum book size efficiently? 
        #    It's hard to know the new max without recomputing or maintaining a heap.
        
        # Given n=40, m=401, recomputing book sizes is O(m * n) = 401 * 40 = 16040 operations.
        # In Python, this is very fast (< 0.01s). So we can afford to recompute in each iteration.
        # 10000 iterations would take < 100s.
        
        no_improve_count = 0
        last_time = time.time()
        
        for i in range(max_iter):
            if time.time() - last_time > 1.0:
                last_time = time.time()
                if time_limit_check():
                    break
            
            # Pick a random edge to remove
            u, v = random.choice(list(edges_set))
            
            # Pick a random non-edge to add
            # Sample until we find a non-edge
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                x, y = min(x, y), max(x, y)
                if (x, y) not in edges_set and (u, v) != (x, y):
                    break
            
            # Perform swap
            edges_set.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
            
            edges_set.add((x, y))
            adj[x].add(y)
            adj[y].add(x)
            
            # Recompute books
            book = compute_books(adj, edges_set)
            new_max = max_book(book)
            
            if new_max < current_max:
                current_max = new_max
                no_improve_count = 0
                if new_max < best_max:
                    best_max = new_max
                    best_edges = list(edges_set)
            else:
                no_improve_count += 1
                # Accept with some probability to escape local minima? 
                # Or just revert if no improvement? 
                # For now, if no improvement, we revert to keep the search focused on minimization.
                # But this might get stuck. Let's try simulated annealing or just random restarts.
                # Given the time, simple hill climbing with random restarts is robust.
                
                # Revert
                edges_set.remove((x, y))
                adj[x].remove(y)
                adj[y].remove(x)
                
                edges_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
            if no_improve_count > 200:
                break
        
        return best_edges, best_max
    
    # Strategy:
    # 1. Generate several random graphs.
    # 2. Run local search on each.
    # 3. Keep the best one.
    
    best_global_max = float('inf')
    best_global_edges = None
    
    num_restarts = 10
    max_iter_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        edges = generate_random_graph(n, m)
        best_edges, best_max = local_search(edges, max_iter_per_restart, lambda: time.time() - start_time > time_limit - 5)
        
        if best_max < best_global_max:
            best_global_max = best_max
            best_global_edges = best_edges
            # Early stop if we reach the theoretical lower bound?
            # The lower bound is 40/6 ~ 6.67, so 7 is the integer lower bound.
            if best_max <= 7:
                break
    
    # If we didn't find anything good, fall back to the seed
    if best_global_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return best_global_edges

# To test locally, we can uncomment the following
# if __name__ == "__main__":
#     edges = construct()
#     n = 40
#     adj = [set() for _ in range(n)]
#     for u, v in edges:
#         adj[u].add(v)
#         adj[v].add(u)
#     book = {}
#     for u, v in edges:
#         if u > v:
#             u, v = v, u
#         common = adj[u] & adj[v]
#         book[(u, v)] = len(common)
#     print("Max book size:", max(book.values()))
#     print("Number of edges:", len(edges))
