import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (150s limit, 2s buffer)
    
    n = 40
    m = 401
    
    # Generate Paley graph on 41 vertices
    residues = {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 31, 32, 33, 36, 37, 39, 40}
    
    adj_41 = [[False]*41 for _ in range(41)]
    for u in range(41):
        for v in range(u+1, 41):
            diff = (v - u) % 41
            if diff in residues:
                adj_41[u][v] = True
                adj_41[v][u] = True
    
    # Remove vertex 0 to get 40 vertices (new vertex i = old vertex i+1)
    adj_mask = [0] * n
    edge_set = set()
    
    for i in range(n):
        old_u = i + 1
        for j in range(i+1, n):
            old_v = j + 1
            if adj_41[old_u][old_v]:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_set.add((i, j))
    
    # Degree balancing: try to make degrees as uniform as possible
    # Current degree count
    def get_degrees(adj_mask):
        return [adj_mask[i].bit_count() for i in range(n)]
    
    def balance_degrees(adj_mask, edge_set, max_iter=500):
        """Balance degrees via edge moves."""
        for _ in range(max_iter):
            degrees = get_degrees(adj_mask)
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            
            # Find an edge (max_idx, u) where u is not connected to min_idx
            moved = False
            mask = adj_mask[max_idx]
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                mask ^= lsb
                if u == min_idx:
                    continue
                if not (adj_mask[min_idx] & (1 << u)):
                    # Move edge (max_idx, u) -> (min_idx, u)
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edge_set.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edge_set.add((min(min_idx, u), max(min_idx, u)))
                    moved = True
                    break
            if not moved:
                break
    
    balance_degrees(adj_mask, edge_set)
    
    # Count edges
    edge_count = len(edge_set)
    # Need exactly 401 edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [e for e in all_edges if e not in edge_set]
    
    # Add or remove edges to get exactly 401
    while edge_count < m:
        # Add edge with minimum common neighbors
        best_e = None
        best_c = float('inf')
        for e in non_edges:
            u, v = e
            c = (adj_mask[u] & adj_mask[v]).bit_count()
            if c < best_c:
                best_c = c
                best_e = e
        if best_e is None:
            break
        u, v = best_e
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add(best_e)
        non_edges.remove(best_e)
        edge_count += 1
    
    while edge_count > m:
        # Remove edge with maximum book size
        books = compute_books_fast(adj_mask)
        max_edge = max(books, key=lambda e: books[e])
        u, v = max_edge
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edge_set.discard(max_edge)
        non_edges.append(max_edge)
        edge_count -= 1
    
    def compute_books_fast(adj_mask):
        """Compute book sizes using bit_count for speed."""
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    books[(u, v)] = (adj_mask[u] & adj_mask[v]).bit_count()
        return books
    
    def get_metrics(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_book, sum_sq
    
    books = compute_books_fast(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # Simulated Annealing with improved acceptance
    T = 3.0
    T_min = 0.05
    cooling = 0.99985
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove - biased toward high-book edges
        if random.random() < 0.75:
            # Top 20% edges by book size
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 5)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
        
        u, v = e_remove
        
        if not non_edges:
            break
        
        # Select edge to add - prefer low common neighbors
        sample_size = min(80, len(non_edges))
        candidates = random.sample(non_edges, sample_size)
        
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            if score < best_add_score:
                best_add_score = score
                best_add = e
        
        e_add = best_add
        
        if e_add is None or e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
        
        a, b = e_add
        
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.discard(e_remove)
        edge_set.add(e_add)
        
        non_edges.remove(e_add)
        non_edges.append(e_remove)
        
        new_books = compute_books_fast(adj_mask)
        new_max_book, new_sum_sq = get_metrics(new_books)
        
        # Acceptance: use smooth surrogate
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            delta_sq = new_sum_sq - current_sum_sq
            if delta_sq <= 0:
                accept = True
            else:
                prob = math.exp(-delta_sq / (T * 100))
                if random.random() < prob:
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
            
            edge_set.discard(e_add)
            edge_set.add(e_remove)
            
            non_edges.remove(e_remove)
            non_edges.append(e_add)
        
        T *= cooling
    
    # Final polish: try local improvements with exact evaluation
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            books = compute_books_fast(best_adj_mask)
            max_book = max(books.values()) if books else 0
            sum_sq = sum(b * b for b in books.values())
            
            # Try swapping each high-book edge with a low-common-neighbor non-edge
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            for e_remove, b_val in sorted_books[:20]:
                if b_val < max_book:
                    break
                u, v = e_remove
                # Find best replacement
                best_add = None
                best_score = float('inf')
                for e in non_edges[:200]:
                    a, b = e
                    score = (best_adj_mask[a] & best_adj_mask[b]).bit_count()
                    if score < best_score:
                        best_score = score
                        best_add = e
                
                if best_add:
                    a, b = best_add
                    # Simulate swap
                    new_mask = list(best_adj_mask)
                    new_mask[u] &= ~(1 << v)
                    new_mask[v] &= ~(1 << u)
                    new_mask[a] |= (1 << b)
                    new_mask[b] |= (1 << a)
                    
                    new_books = compute_books_fast(new_mask)
                    new_max = max(new_books.values()) if new_books else 0
                    new_sq = sum(x * x for x in new_books.values())
                    
                    if new_max < max_book or (new_max == max_book and new_sq < sum_sq):
                        best_adj_mask = new_mask
                        best_edge_set.discard(e_remove)
                        best_edge_set.add(best_add)
                        non_edges.remove(best_add)
                        non_edges.append(e_remove)
                        improved = True
                        break
    
    return [list(e) for e in best_edge_set]
