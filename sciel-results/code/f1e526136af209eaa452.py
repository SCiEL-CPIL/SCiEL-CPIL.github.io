import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = common.bit_count()
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    # Initialization: Structured graph
    # Partition: L1(0-9), L2(10-19), R1(20-29), R2(30-39)
    # Add internal cliques K10
    groups = [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40))
    ]
    
    edge_set = set()
    
    # Add internal edges
    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                u, v = g[i], g[j]
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
    # Current edge count
    current_edges = len(edge_set)
    # Need m - current_edges more edges
    # These should be bipartite edges between {L1, L2} and {R1, R2}
    # L = 0-19, R = 20-39
    
    left_verts = list(range(0, 20))
    right_verts = list(range(20, 40))
    
    # Generate all possible bipartite edges
    bipartite_edges = [(u, v) for u in left_verts for v in right_verts]
    random.shuffle(bipartite_edges)
    
    needed = m - current_edges
    # Add 'needed' random bipartite edges
    for e in bipartite_edges[:needed]:
        u, v = e
        edge_set.add(e)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Now we have 401 edges
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    current_sum_sq = get_sum_sq_books(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9999
    
    # Precompute list of all possible edges for quick non-edge sampling
    # Actually, we can just sample from all possible and check if in edge_set
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Bias towards high book size
        if random.random() < 0.8:
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 10)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Select edge to add
        # Bias towards low common neighbors
        # Sample 50 random non-edges
        non_edges = [e for e in all_possible if e not in edge_set]
        if not non_edges:
            break
            
        candidates = random.sample(non_edges, min(50, len(non_edges)))
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            if score < best_add_score:
                best_add_score = score
                best_add = e
                
        e_add = best_add
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
            
        a, b = e_add
        
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        # Recompute books
        # Optimization: Only recompute affected edges?
        # For simplicity and correctness, recompute all.
        # With n=40, m=401, this is fast enough.
        new_books = compute_books(adj_mask)
        new_max_book = get_max_book(new_books)
        new_sum_sq = get_sum_sq_books(new_books)
        
        # Acceptance
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                delta_sq = new_sum_sq - current_sum_sq
                if delta_sq > 0:
                    prob = math.exp(-delta_sq / (T * 100))
                    if random.random() < prob:
                        accept = True
                else:
                    accept = True
        else:
            delta_max = new_max_book - current_max_book
            prob = math.exp(-delta_max / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            books = new_books
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
