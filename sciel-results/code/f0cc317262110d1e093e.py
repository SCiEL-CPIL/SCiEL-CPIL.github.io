import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Helper to compute max book size using numpy for speed
    def get_max_book(edge_list):
        if not edge_list:
            return 0
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edge_list:
            adj[u, v] = 1
            adj[v, u] = 1
        
        max_book = 0
        for u, v in edge_list:
            # Count common neighbors
            count = np.sum(adj[u] & adj[v])
            if count > max_book:
                max_book = count
        return max_book

    # Helper to get book sizes for all edges as a dict
    def get_book_sizes(edge_set):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
        
        books = {}
        for u, v in edge_set:
            count = np.sum(adj[u] & adj[v])
            books[(u, v)] = count
        return books, adj

    # Structural initialization: 3-partite graph T3(14, 13, 13)
    # Parts: 0..13, 14..26, 27..39
    # Edges: all between different parts.
    # Total edges = 14*13 + 14*13 + 13*13 = 182 + 182 + 169 = 533.
    # We need to remove 533 - 401 = 132 edges.
    
    part0 = list(range(0, 14))
    part1 = list(range(14, 27))
    part2 = list(range(27, 40))
    
    def create_initial_graph(seed_val):
        random.seed(seed_val)
        edges = set()
        
        # Add all T3 edges
        for u in part0:
            for v in part1:
                edges.add((u, v))
            for v in part2:
                edges.add((u, v))
        for u in part1:
            for v in part2:
                edges.add((u, v))
        
        # We need to remove 132 edges.
        # Strategy: Randomly remove edges, but prioritize removing edges that are in many triangles?
        # Or just random removal to break symmetry?
        # Let's try random removal first, then optimize.
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        # Remove first 132 edges
        for e in edge_list[:132]:
            edges.remove(e)
            
        return edges

    def optimize(edges_set, time_limit):
        current_edges = set(edges_set)
        books, adj = get_book_sizes(current_edges)
        current_max = max(books.values()) if books else 0
        
        best_edges = set(current_edges)
        best_max = current_max
        
        # Local search: Swap edges
        # Pick a high-book edge, remove it, add a random non-edge.
        # Accept if max book decreases or stays same (with some probability).
        
        temp = 10.0
        cooling = 0.999
        
        while time.time() < time_limit:
            if not current_edges:
                break
                
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
            # To maintain structure, maybe pick an edge from a random pair of non-adjacent vertices
            # Or just random pair
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
            
            # Perform swap
            current_edges.remove(worst_edge)
            current_edges.add(new_edge)
            
            # Recalculate books (expensive, but n is small)
            # Optimization: Only recalculate books for edges affected by the swap?
            # Edges affected: those incident to u, v, a, b.
            # But for simplicity and correctness, let's just recalculate all if we are fast enough.
            # 401 edges * 40 checks = ~16000 ops. Should be fast in Python? Maybe.
            # Let's try incremental update if possible, or just full recalc.
            
            new_books, new_adj = get_book_sizes(current_edges)
            new_max = max(new_books.values()) if new_books else 0
            
            delta = new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718, -delta/temp):
                current_max = new_max
                books = new_books
                adj = new_adj
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.remove(new_edge)
                current_edges.add(worst_edge)
                
            temp *= cooling
            
        return best_edges, best_max

    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts with different seeds
    seed = 42
    while time.time() < deadline - 10.0:
        initial_edges = create_initial_graph(seed)
        optimized_edges, score = optimize(initial_edges, deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_edges = optimized_edges
            # If we reach theoretical lower bound (7), stop
            if best_score <= 7:
                break
        
        seed += 1

    if best_edges is None:
        # Fallback: K20,20 + 1 edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    # Convert to list of lists
    return [[u, v] for u, v in best_edges]
