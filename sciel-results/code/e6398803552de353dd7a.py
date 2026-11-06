import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = bin(common).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def random_regular_graph(n, d):
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        adj_mask = [0] * n
        edges = set()
        
        chosen = all_possible[:400]
        for u, v in chosen:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.add((u, v))
            
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        for _ in range(200):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
                
            moved = False
            mask = adj_mask[max_idx]
            while mask:
                u = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edges.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edges.add((min(min_idx, u), max(min_idx, u)))
                    
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    moved = True
                    break
            if not moved:
                break
                
        return edges, adj_mask

    edge_set, adj_mask = random_regular_graph(n, 20)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_possible if e not in edge_set]
    random.shuffle(missing)
    if missing:
        e_add = missing[0]
        u, v = e_add
        edge_set.add(e_add)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    
    best_max_book = current_max_book
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9999
    
    while time.time() < deadline - 5.0 and T > T_min:
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        top_k = max(1, len(sorted_books) // 5)
        e_remove = random.choice(sorted_books[:top_k])[0]
        u, v = e_remove
        
        candidates = []
        for _ in range(20):
            e = random.choice(all_possible)
            if e not in edge_set:
                candidates.append(e)
                
        if not candidates:
            T *= cooling
            continue
            
        best_cand = None
        best_score = current_max_book + 10
        
        for e in candidates:
            a, b = e
            new_book_ab = bin(adj_mask[a] & adj_mask[b]).count('1')
            if new_book_ab < best_score:
                best_score = new_book_ab
                best_cand = e
                
        if not best_cand:
            T *= cooling
            continue
            
        a, b = best_cand
        
        if best_cand == e_remove:
            continue
            
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(best_cand)
        
        books = compute_books(adj_mask)
        new_max_book = get_max_book(books)
        
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            accept = True
        else:
            if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                accept = True
        if accept:
            current_max_book = new_max_book
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove(best_cand)
            edge_set.add(e_remove)
            
            books = compute_books(adj_mask)
            current_max_book = get_max_book(books)
            
        T *= cooling
        
    # Polish Phase
    # Deterministic local search
    # Find max book edge, try all swaps, pick best.
    # Repeat until no improvement or time limit.
    
    while time.time() < deadline - 1.0:
        books = compute_books(adj_mask)
        current_max_book = get_max_book(books)
        
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            
        # Find edge with max book
        max_book = current_max_book
        target_edges = [e for e, b in books.items() if b == max_book]
        
        if not target_edges:
            break
            
        e_remove = random.choice(target_edges)
        u, v = e_remove
        
        best_swap = None
        best_new_max = max_book
        
        # Try all non-edges
        # This is 379 non-edges.
        # For each, compute new max book?
        # That's 379 * 400 = 150k ops per target edge.
        # If we have 5 target edges, 750k ops.
        # This might be too slow if done many times.
        # Let's limit the polish iterations or optimize.
        
        # Optimization: Only try a subset of non-edges?
        # Or just do it once per loop?
        
        non_edges = [e for e in all_possible if e not in edge_set]
        
        for e_add in non_edges:
            a, b = e_add
            if e_add == e_remove:
                continue
                
            # Simulate
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            # Compute max book
            # We only need to check edges affected by u, v, a, b?
            # No, removing u-v might reduce book of u-v, but adding a-b might increase others.
            # Actually, only edges incident to u, v, a, b change book size.
            # Plus the new edge a-b.
            
            # Let's just compute all books for simplicity in polish (slower but robust)
            # Or compute only affected?
            
            # Affected vertices: u, v, a, b
            # Edges to check:
            # 1. All edges incident to u, v, a, b
            # 2. The new edge (a,b)
            
            max_val = 0
            # We need the current books for unaffected edges to know the global max?
            # No, we want the NEW global max.
            # Unaffected edges have same book size.
            # So new_max = max( max_book_unaffected, max_book_affected_new )
            
            # We need max_book_unaffected.
            # This is max(books[e] for e in books if e is not incident to u,v,a,b)
            
            # This is getting complex. Let's just recompute all books.
            # 400 edges. Fast.
            
            new_books = {}
            for i in range(n):
                mask = adj_mask[i]
                while mask:
                    j = (mask & -mask).bit_length() - 1
                    mask &= mask - 1
                    if i < j:
                        common = adj_mask[i] & adj_mask[j]
                        new_books[(i, j)] = bin(common).count('1')
                        
            new_max = max(new_books.values()) if new_books else 0
            
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_swap = e_add
                
        if best_swap:
            a, b = best_swap
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            edge_set.remove(e_remove)
            edge_set.add(best_swap)
            
            # Update books
            books = compute_books(adj_mask)
        else:
            break
            
    return [list(e) for e in best_edge_set]
