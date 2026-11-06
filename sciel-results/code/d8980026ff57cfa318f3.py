import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    target_edges = 401
    
    def calc_max_book(A):
        books = A @ A
        return int(np.max(books * A))
    
    def get_books(A):
        return A @ A

    best_max_book = n * n
    best_A = None
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        random.shuffle(all_pairs)
        initial_edges = all_pairs[:target_edges]
        
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in initial_edges:
            A[u, v] = 1
            A[v, u] = 1
            
        cur_max = calc_max_book(A)
        
        if cur_max < best_max_book:
            best_max_book = cur_max
            best_A = A.copy()
            
        if best_max_book <= 7:
            break
            
        # Local Search
        stuck = 0
        while time.time() < deadline - 5:
            if stuck > 100:
                break
                
            books = get_books(A)
            cur_max = int(np.max(books * A))
            
            # Find edge with max book
            flat_books = (books * A).flatten()
            max_val = flat_books.max()
            if max_val <= 7:
                break
                
            # Find all edges with max book
            max_idx = np.argmax(flat_books)
            u, v = np.unravel_index(max_idx, (n, n))
            
            # Find best non-edge to add
            non_edge_mask = (A == 0)
            cand_books = books * non_edge_mask
            np.fill_diagonal(cand_books, np.inf)
            
            min_idx = np.argmin(cand_books)
            x, y = np.unravel_index(min_idx, (n, n))
            
            # Perform swap
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_max = calc_max_book(A)
            
            if new_max < cur_max:
                cur_max = new_max
                stuck = 0
                if cur_max < best_max_book:
                    best_max_book = cur_max
                    best_A = A.copy()
                    if best_max_book <= 7:
                        break
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                stuck += 1
                
                # Random restart of local search logic if stuck
                if stuck % 10 == 0:
                    non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
                    if non_edges_list:
                        rx, ry = random.choice(non_edges_list)
                        A[u, v] = 0
                        A[v, u] = 0
                        A[rx, ry] = 1
                        A[ry, rx] = 1
                        
                        new_max_rand = calc_max_book(A)
                        if new_max_rand < cur_max:
                            cur_max = new_max_rand
                            stuck = 0
                            if cur_max < best_max_book:
                                best_max_book = cur_max
                                best_A = A.copy()
                                if best_max_book <= 7:
                                    break
                        else:
                            A[u, v] = 1
                            A[v, u] = 1
                            A[rx, ry] = 0
                            A[ry, rx] = 0

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    # Ensure exactly 401
    if len(edges) != target_edges:
        existing = set(tuple(e) for e in edges)
        if len(edges) < target_edges:
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
        else:
            edges = edges[:target_edges]
            
    return edges
