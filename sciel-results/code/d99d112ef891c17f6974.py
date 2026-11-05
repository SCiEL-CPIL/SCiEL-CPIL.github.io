<<<<<<< SEARCH
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Partition vertices into 3 parts: 14, 13, 13
    part0 = list(range(14))
    part1 = list(range(14, 27))
    part2 = list(range(27, 40))
    
    best_score = float('inf')
    best_edges = None
    
    # Function to calculate max book size
    def get_max_book(edges):
        if not edges:
            return 0
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        max_book = 0
        for u, v in edges:
            # Count common neighbors
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            if count > max_book:
                max_book = count
        return max_book

    # Greedy removal from T3
    def greedy_remove(seed_val):
        random.seed(seed_val)
        
        # Initialize T3 edges
        edges = set()
        for u in part0:
            for v in part1:
                edges.add((u, v))
            for v in part2:
                edges.add((u, v))
        for u in part1:
            for v in part2:
                edges.add((u, v))
                
        edge_list = list(edges)
        
        # Build adjacency matrix for fast lookup
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Calculate initial book sizes
        book = [[0]*n for _ in range(n)]
        for u, v in edges:
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            book[u][v] = count
            book[v][u] = count
            
        current_edges = len(edges)
        
        # We need to remove until we have target_edges
        while current_edges > target_edges:
            # Find edge with max book size
            max_b = -1
            candidates = []
            for u, v in edges:
                b = book[u][v]
                if b > max_b:
                    max_b = b
                    candidates = [(u, v)]
                elif b == max_b:
                    candidates.append((u, v))
            
            # Randomly pick one from candidates
            u, v = random.choice(candidates)
            
            # Remove edge (u, v)
            edges.remove((u, v))
            adj[u][v] = False
            adj[v][u] = False
            
            # Update book sizes for neighbors
            # Neighbors of (u, v) are edges (u, w) and (v, w) where w is common neighbor
            # Actually, any edge (u, w) or (v, w) that shares a triangle with (u, v)
            # Triangle (u, v, w) exists if adj[u][w] and adj[v][w]
            # Removing (u, v) reduces book[u][w] and book[v][w] by 1 for all such w
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    # (u, w) and (v, w) are in the graph
                    # Check if they are in edges set
                    if (u, w) in edges or (w, u) in edges:
                         # Find canonical form
                         e1 = (u, w) if u < w else (w, u)
                         if e1 in edges:
                             # Update book for e1
                             # book[e1[0]][e1[1]] -= 1
                             pass # We need to update the matrix correctly
                    # Simpler: just iterate through all edges incident to u or v?
                    # No, just update book[u][w] and book[v][w]
                    # But we need to ensure (u,w) and (v,w) are valid edges
                    if (u, w) in edges:
                        book[u][w] -= 1
                    elif (w, u) in edges:
                        book[w][u] -= 1
                    if (v, w) in edges:
                        book[v][w] -= 1
                    elif (w, v) in edges:
                        book[w][v] -= 1
                            
            current_edges -= 1
            
            # Time check
            if time.time() > deadline - 5.0:
                break
                
        # Convert to list of lists
        res = [[u, v] for u, v in edges]
        return res

    # Run multiple restarts
    iterations = 0
    while time.time() < deadline - 10.0:
        edges = greedy_remove(iterations)
        score = get_max_book(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            # If we hit the theoretical lower bound, we can stop
            if best_score <= 7:
                break
        iterations += 1
        
    if best_edges is None:
        # Fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
=======
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    def get_max_book(adj):
        """adj is n x n boolean numpy array. Returns max book size."""
        # book_size[i,j] = number of common neighbors of i and j = sum_k adj[i,k] & adj[j,k]
        # For each edge (i,j), book = dot(adj[i], adj[j])
        max_book = 0
        # Only check edges that exist
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = np.sum(adj[i] & adj[j])
                    if common > max_book:
                        max_book = common
        return max_book

    def get_max_book_edges(adj, edges):
        """Faster: only check the given edges."""
        max_book = 0
        for u, v in edges:
            common = np.sum(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book

    def initial_graph(seed_val):
        """Create an initial graph with exactly target_edges edges."""
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        # Start with K_{20,20} plus one edge (like the seed program)
        adj = np.zeros((n, n), dtype=bool)
        edges = set()
        
        for i in range(20):
            for j in range(20, 40):
                adj[i, j] = True
                adj[j, i] = True
                edges.add((i, j))
        
        # Add one more edge within a part
        adj[0, 1] = True
        adj[1, 0] = True
        edges.add((0, 1))
        
        return adj, list(edges)

    def local_search(adj, edges, seed_val, time_limit):
        """Perform local search: try edge swaps (remove one, add another) to reduce max book."""
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        current_score = get_max_book_edges(adj, edges)
        best_score = current_score
        best_adj = adj.copy()
        best_edges = list(edges)
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Pick a random edge to remove
            u, v = random.choice(edges)
            
            # Pick a random non-edge to add
            # Try several random non-edges
            added = False
            for _ in range(50):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                a, b = min(a,b), max(a,b)
                if (a, b) == (u, v):
                    continue
                if not adj[a, b]:
                    # Try the swap
                    adj[u, v] = False
                    adj[v, u] = False
                    edges.remove((u, v))
                    
                    adj[a, b] = True
                    adj[b, a] = True
                    edges.add((a, b))
                    
                    new_score = get_max_book_edges(adj, edges)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj.copy()
                        best_edges = list(edges)
                        no_improve_count = 0
                    else:
                        # Revert
                        adj[a, b] = False
                        adj[b, a] = False
                        edges.remove((a, b))
                        
                        adj[u, v] = True
                        adj[v, u] = True
                        edges.add((u, v))
                        no_improve_count += 1
                    
                    added = True
                    break
            
            if not added:
                no_improve_count += 1
        
        return best_score, best_edges

    def local_search_targeted(adj, edges, seed_val, time_limit):
        """Local search that specifically targets high-book edges for removal."""
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        current_score = get_max_book_edges(adj, edges)
        best_score = current_score
        best_adj = adj.copy()
        best_edges = list(edges)
        
        no_improve_count = 0
        max_no_improve = 300
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Find edges with highest book size
            edge_books = []
            for u, v in edges:
                common = np.sum(adj[u] & adj[v])
                edge_books.append((common, u, v))
            
            edge_books.sort(reverse=True)
            
            # Pick from top candidates (high book size edges)
            top_k = min(10, len(edge_books))
            if top_k > 0:
                _, u, v = random.choice(edge_books[:top_k])
            else:
                u, v = random.choice(edges)
            
            # Try to remove this edge and add a random non-edge
            added = False
            for _ in range(100):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                a, b = min(a,b), max(a,b)
                if (a, b) == (u, v):
                    continue
                if not adj[a, b]:
                    # Perform swap
                    adj[u, v] = False
                    adj[v, u] = False
                    edges.remove((u, v))
                    
                    adj[a, b] = True
                    adj[b, a] = True
                    edges.add((a, b))
                    
                    new_score = get_max_book_edges(adj, edges)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj.copy()
                        best_edges = list(edges)
                        no_improve_count = 0
                    else:
                        # Revert
                        adj[a, b] = False
                        adj[b, a] = False
                        edges.remove((a, b))
                        
                        adj[u, v] = True
                        adj[v, u] = True
                        edges.add((u, v))
                        no_improve_count += 1
                    
                    added = True
                    break
            
            if not added:
                no_improve_count += 1
        
        return best_score, best_edges

    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts with different initializations and search strategies
    iterations = 0
    while time.time() < deadline - 15.0:
        # Alternate between different initialization strategies
        if iterations % 3 == 0:
            # Standard K_{20,20} + 1 edge
            adj, edges = initial_graph(iterations)
            score, new_edges = local_search(adj, edges, iterations, deadline - 10.0)
        elif iterations % 3 == 1:
            # Targeted local search
            adj, edges = initial_graph(iterations)
            score, new_edges = local_search_targeted(adj, edges, iterations, deadline - 10.0)
        else:
            # Different initial structure: start with random-ish graph
            random.seed(iterations)
            adj = np.zeros((n, n), dtype=bool)
            edges = set()
            
            # Generate a random graph with target_edges edges
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_possible)
            for e in all_possible[:target_edges]:
                u, v = e
                adj[u, v] = True
                adj[v, u] = True
                edges.add(e)
            
            score, new_edges = local_search_targeted(adj, list(edges), iterations, deadline - 10.0)
        
        if score < best_score:
            best_score = score
            best_edges = new_edges
            if best_score <= 7:
                break
        
        iterations += 1
    
    if best_edges is None:
        # Fallback
        adj, edges = initial_graph(0)
        best_edges = edges
    
    return best_edges
>>>>>>> REPLACE
