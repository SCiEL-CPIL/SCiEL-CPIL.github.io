import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    adj_mask = [0] * n
    edge_set = set()
    books = {}
    book_counts = [0] * (n + 1) # book_counts[k] = number of edges with book size k
    max_book = 0
    sum_sq = 0
    
    def update_books_for_edge(u, v):
        """Update books and histogram for a single edge (u,v). 
        Assumes (u,v) is in edge_set or we are calculating its potential book size."""
        common = adj_mask[u] & adj_mask[v]
        b = common.bit_count()
        return b

    def add_edge(u, v):
        nonlocal edge_set, adj_mask, books, book_counts, max_book, sum_sq
        u, v = min(u, v), max(u, v)
        if (u, v) in edge_set:
            return
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        b = update_books_for_edge(u, v)
        books[(u, v)] = b
        book_counts[b] += 1
        sum_sq += b * b
        if b > max_book:
            max_book = b

    def remove_edge(u, v):
        nonlocal edge_set, adj_mask, books, book_counts, max_book, sum_sq
        u, v = min(u, v), max(u, v)
        if (u, v) not in edge_set:
            return
        edge_set.remove((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        b = books.pop((u, v))
        book_counts[b] -= 1
        sum_sq -= b * b
        # max_book might decrease, we'll handle that in the main loop if needed
        # or just let it be and check later. Actually, if we remove the max, we need to find new max.
        
    def recalc_max_book():
        nonlocal max_book
        # Find largest k with book_counts[k] > 0
        for k in range(max_book, -1, -1):
            if book_counts[k] > 0:
                max_book = k
                return
        max_book = 0

    def get_affected_edges(u, v, a, b):
        """Get set of edges whose book size might change after swapping (u,v) -> (a,b)."""
        verts = {u, v, a, b}
        affected = set()
        
        for w in verts:
            # Edges incident to w
            mask = adj_mask[w]
            while mask:
                z = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                affected.add((min(w, z), max(w, z)))
            
            # Edges (x, y) where w is a common neighbor
            # i.e., x, y in N(w) and (x, y) in edge_set
            neigh = []
            mask = adj_mask[w]
            while mask:
                z = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                neigh.append(z)
                
            for i in range(len(neigh)):
                for j in range(i+1, len(neigh)):
                    x, y = neigh[i], neigh[j]
                    if x > y: x, y = y, x
                    if (x, y) in edge_set:
                        affected.add((x, y))
        
        return affected

    def greedy_randomized_init(n, m):
        nonlocal adj_mask, edge_set, books, book_counts, max_book, sum_sq
        adj_mask = [0] * n
        edge_set = set()
        books = {}
        book_counts = [0] * (n + 1)
        max_book = 0
        sum_sq = 0
        
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        
        # Start with random subset
        # Use GRASP
        set_a = list(range(20))
        set_b = list(range(20, 40))
        bipartite_edges = [(u, v) for u in set_a for v in set_b]
        random.shuffle(bipartite_edges)
        
        num_bipartite = random.randint(350, 390)
        for u, v in bipartite_edges[:num_bipartite]:
            add_edge(u, v)
        
        remaining_needed = m - len(edge_set)
        all_candidates = [e for e in all_possible if e not in edge_set]
        random.shuffle(all_candidates)
        
        for _ in range(remaining_needed):
            if not all_candidates:
                break
            scored_candidates = []
            for e in all_candidates[:min(50, len(all_candidates))]:
                u, v = e
                common = (adj_mask[u] & adj_mask[v]).bit_count()
                scored_candidates.append((common, e))
            
            if not scored_candidates:
                e = random.choice(all_candidates)
                add_edge(e[0], e[1])
                all_candidates.remove(e)
                continue
            
            scored_candidates.sort(key=lambda x: x[0])
            rcl_size = max(1, len(scored_candidates) // 3)
            rcl = scored_candidates[:rcl_size]
            chosen_book, chosen_edge = random.choice(rcl)
            add_edge(chosen_edge[0], chosen_edge[1])
            if chosen_edge in all_candidates:
                all_candidates.remove(chosen_edge)
        
        recalc_max_book()

    # Initialize
    greedy_randomized_init(n, m)
    
    best_max_book = max_book
    best_sum_sq = sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    best_books = dict(books)
    best_book_counts = list(book_counts)

    T = 2.0
    T_min = 0.05
    cooling = 0.9998
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline and T > T_min:
        if random.random() < 0.85:
            # Pick from top book size
            top_k = max(1, len(edge_set) // 7)
            # We need to pick an edge with high book size.
            # We can iterate through book_counts to find the bucket.
            # Or just sample edges until we find a good one?
            # Better: pick a random edge, if its book size is not max, try again?
            # Or maintain a list of edges by book size?
            # For now, just pick random and hope, or pick from top bucket.
            
            # Let's pick an edge from the current max_book bucket if possible
            candidates = [e for e in edge_set if books.get(e, 0) >= max_book - 1]
            if not candidates:
                e_remove = random.choice(list(edge_set))
            else:
                e_remove = random.choice(candidates)
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Find best replacement
        best_cand = None
        best_score = None
        
        # Sample non-edges
        for _ in range(30):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                # Simulate swap
                # We need to estimate the new max_book and sum_sq
                
                # This is expensive to do exactly for every candidate.
                # Approximation: 
                # 1. Remove (u,v). 
                # 2. Add (a,b).
                # 3. Check max book.
                
                # To speed up, we can just check the book size of (a,b) 
                # and the neighbors.
                
                # Let's do a quick check:
                # Save state
                # Apply
                # Compute max
                # Revert
                
                # This is 30 * (cost of apply/revert).
                # Cost of apply/revert is ~100 ops.
                # 3000 ops per iteration.
                # 140s / (3000 ops / 10M ops/s) = 140s / 0.3ms = 460k iterations.
                # This is feasible.
                
                # But we need to be careful with the state.
                
                # Let's just pick the candidate with the smallest book size for (a,b)
                # and hope it doesn't create a new max.
                common_ab = (adj_mask[a] & adj_mask[b]).bit_count()
                
                # Heuristic score: common_ab + penalty if it creates high book
                # If common_ab < max_book, it's good.
                
                if best_cand is None or common_ab < best_score:
                    best_cand = (a, b)
                    best_score = common_ab
                    
        if best_cand is None:
            T *= cooling
            continue
            
        a, b = best_cand
        
        # Perform the swap
        # 1. Identify affected edges
        affected = get_affected_edges(u, v, a, b)
        
        # 2. Save old books for affected edges
        old_books = {e: books[e] for e in affected if e in books}
        
        # 3. Remove (u,v)
        remove_edge(u, v)
        
        # 4. Add (a,b)
        add_edge(a, b)
        
        # 5. Update books for affected edges
        for e in affected:
            if e in edge_set:
                x, y = e
                new_b = update_books_for_edge(x, y)
                if e in books:
                    # It was there before, so we need to update histogram
                    old_b = books[e]
                    book_counts[old_b] -= 1
                    sum_sq -= old_b * old_b
                    books[e] = new_b
                    book_counts[new_b] += 1
                    sum_sq += new_b * new_b
                else:
                    # It's a new edge (a,b)
                    books[e] = new_b
                    book_counts[new_b] += 1
                    sum_sq += new_b * new_b
            else:
                # Edge was removed (u,v)
                # It was already removed from books and histogram in remove_edge
                # But wait, remove_edge only removed (u,v).
                # Other edges in 'affected' that are NOT in edge_set anymore?
                # Only (u,v) is removed.
                pass
        
        # 6. Recalculate max_book
        recalc_max_book()
        
        # 7. Acceptance
        accept = False
        if max_book < best_max_book:
            accept = True
        elif max_book == best_max_book:
            if sum_sq < best_sum_sq:
                accept = True
            else:
                if random.random() < math.exp(-(sum_sq - best_sum_sq) / (T * 300)):
                    accept = True
        else:
            if random.random() < math.exp(-(max_book - best_max_book) / T):
                accept = True
        
        if accept:
            best_max_book = max_book
            best_sum_sq = sum_sq
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            best_books = dict(books)
            best_book_counts = list(book_counts)
            
            if best_max_book == 7:
                return [list(e) for e in best_edge_set]
        else:
            # Revert
            # Remove (a,b)
            remove_edge(a, b)
            # Add (u,v)
            add_edge(u, v)
            
            # Restore books for affected edges
            for e, b_val in old_books.items():
                if e in books:
                    # Update histogram
                    new_b = books[e]
                    book_counts[new_b] -= 1
                    sum_sq -= new_b * new_b
                    books[e] = b_val
                    book_counts[b_val] += 1
                    sum_sq += b_val * b_val
                else:
                    # This case shouldn't happen for old_books
                    pass
            
            # Restore (u,v) book? 
            # add_edge(u,v) added it to books and histogram.
            # But we need to make sure the histogram is consistent.
            # The restore loop above handled edges that were in old_books.
            # (u,v) was in old_books? 
            # affected included (u,v)? 
            # get_affected_edges(u,v,a,b) includes edges incident to u,v.
            # So (u,v) is in affected.
            # So (u,v) was in old_books.
            # So it was restored.
            
            recalc_max_book()
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
