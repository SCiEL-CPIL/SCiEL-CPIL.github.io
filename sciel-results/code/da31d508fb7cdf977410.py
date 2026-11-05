import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
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

    def balanced_bipartite_init(set_a, set_b, extra_edge):
        """
        Initialize with K20,20 plus one specified extra edge.
        """
        adj_mask = [0] * n
        edge_set = set()
        
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        u, v = extra_edge
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask

    def targeted_local_search(edge_set, adj_mask, books, max_steps=5000):
        """
        Greedy + short SA focused on reducing max book.
        Returns (edge_set, adj_mask, books, max_book, sum_sq)
        """
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        T = 1.0
        T_min = 0.1
        cooling = 0.999
        
        step = 0
        while step < max_steps and time.time() < deadline - 5.0 and T > T_min:
            # Build sorted list of edges by book size
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            
            # Select edge to remove: strongly prefer high-book edges
            if current_max_book > 7:
                # Focus on edges with book >= current_max_book
                high_edges = [e for e, b in sorted_books if b >= current_max_book]
                if high_edges:
                    e_remove = random.choice(high_edges)
                else:
                    e_remove = random.choice(sorted_books[:max(1, len(sorted_books)//10)])[0]
            else:
                # Try to reduce sum_sq while keeping max low
                top_k = max(1, len(sorted_books) // 8)
                e_remove = random.choice(sorted_books[:top_k])[0]
            
            u, v = e_remove
            u, v = min(u, v), max(u, v)
            
            # Select edge to add: minimize common neighbors
            # Sample non-edges and pick best
            best_cand = None
            best_common = 100
            best_new_max = 100
            best_new_sum_sq = 10**9
            
            # Try up to 40 candidates
            for _ in range(40):
                a, b = random.choice(all_possible)
                a, b = min(a, b), max(a, b)
                if (a, b) in edge_set or (a, b) == e_remove:
                    continue
                
                # Estimate new book size for this edge
                common = (adj_mask[a] & adj_mask[b]).bit_count()
                
                # Quick check: would this create a worse max?
                # The new edge (a,b) has book size = common
                # Edges involving a or b may change
                if common < best_common:
                    best_common = common
                    best_cand = (a, b)
            
            if best_cand is None:
                T *= cooling
                step += 1
                continue
            
            a, b = best_cand
            
            # Apply the move and compute exact delta
            affected_verts = {u, v, a, b}
            
            # Save old books for affected edges
            old_books = {}
            for e in edge_set:
                x, y = e
                if x in affected_verts or y in affected_verts:
                    old_books[e] = books[e]
            
            # Apply
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            edge_set.remove(e_remove)
            edge_set.add((a, b))
            
            # Recompute books for affected edges
            new_books = {}
            for e in edge_set:
                x, y = e
                if x in affected_verts or y in affected_verts:
                    new_books[e] = (adj_mask[x] & adj_mask[y]).bit_count()
            
            # Update global books
            for e in old_books:
                if e in new_books:
                    books[e] = new_books[e]
                else:
                    del books[e]
            for e in new_books:
                if e not in old_books:
                    books[e] = new_books[e]
            
            new_max_book = get_max_book(books)
            new_sum_sq = get_sum_sq_books(books)
            
            # Acceptance: strictly prefer lower max, then lower sum_sq
            accept = False
            if new_max_book < current_max_book:
                accept = True
            elif new_max_book == current_max_book:
                if new_sum_sq < current_sum_sq:
                    accept = True
                elif random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 200)):
                    accept = True
            else:
                # Only accept if we're at high temperature and the increase is small
                if new_max_book <= current_max_book + 1 and random.random() < math.exp(-(new_max_book - current_max_book) / T):
                    accept = True
            
            if accept:
                current_max_book = new_max_book
                current_sum_sq = new_sum_sq
            else:
                # Revert
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                
                edge_set.remove((a, b))
                edge_set.add(e_remove)
                
                for e in new_books:
                    if e in old_books:
                        books[e] = old_books[e]
                    else:
                        del books[e]
                for e in old_books:
                    if e not in new_books:
                        books[e] = old_books[e]
            
            T *= cooling
            step += 1
        
        return edge_set, adj_mask, books, current_max_book, current_sum_sq

    # Portfolio: try multiple diverse starting points
    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    set_a = list(range(20))
    set_b = list(range(20, 40))
    
    # Generate diverse extra edges
    extra_edges = []
    # Within set_a
    for i in range(0, 18, 3):
        for j in range(i+1, 20, 3):
            extra_edges.append((i, j))
    # Within set_b
    for i in range(20, 38, 3):
        for j in range(i+1, 40, 3):
            extra_edges.append((i, j))
    # Cross edges that are already in K20,20 - skip those
    # Add some random ones
    for _ in range(20):
        if random.random() < 0.5:
            u, v = random.sample(set_a, 2)
        else:
            u, v = random.sample(set_b, 2)
        extra_edges.append((min(u,v), max(u,v)))
    
    # Deduplicate
    extra_edges = list(set(extra_edges))
    random.shuffle(extra_edges)
    
    # Try up to 15 diverse starts
    num_starts = min(15, len(extra_edges))
    for idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        extra = extra_edges[idx]
        edge_set, adj_mask = balanced_bipartite_init(set_a, set_b, extra)
        books = compute_books(adj_mask)
        
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        # Run targeted local search
        edge_set, adj_mask, books, current_max_book, current_sum_sq = targeted_local_search(
            edge_set, adj_mask, books, max_steps=3000
        )
        
        if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
            best_max_book = current_max_book
            best_sum_sq = current_sum_sq
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            
            if best_max_book == 7:
                return [list(e) for e in best_edge_set]
    
    # If we have time, do a final longer SA from the best solution
    if time.time() < deadline - 15.0 and best_edge_set is not None:
        edge_set = set(best_edge_set)
        adj_mask = list(best_adj_mask)
        books = compute_books(adj_mask)
        
        # Run a longer SA
        edge_set, adj_mask, books, final_max, final_sum_sq = targeted_local_search(
            edge_set, adj_mask, books, max_steps=10000
        )
        
        if final_max < best_max_book or (final_max == best_max_book and final_sum_sq < best_sum_sq):
            best_max_book = final_max
            best_sum_sq = final_sum_sq
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
    
    return [list(e) for e in best_edge_set]
