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
    
    def compute_books_full(adj_mask):
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
        if not books: return 0
        return max(books.values())

    def init_bipartite_plus_one():
        adj_mask = [0] * n
        edge_set = set()
        
        set_a = list(range(20))
        set_b = list(range(20, 40))
        
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Add 1 random edge within a set
        if random.random() < 0.5:
            u, v = random.sample(set_a, 2)
        else:
            u, v = random.sample(set_b, 2)
        
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask

    # Try multiple initializations
    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    
    for _ in range(5):
        edge_set, adj_mask = init_bipartite_plus_one()
        books = compute_books_full(adj_mask)
        current_max_book = get_max_book(books)
        
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            best_books = books # Keep track
            
        if time.time() > deadline - 10:
            break

    edge_set = best_edge_set
    adj_mask = best_adj_mask
    books = best_books
    
    # Incremental Local Search
    # Target: Reduce max book to 7
    
    max_iterations = 10000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        current_max_book = get_max_book(books)
        if current_max_book <= 7:
            break
        
        # Find an edge with max book
        max_edges = [e for e, b in books.items() if b == current_max_book]
        if not max_edges:
            break
            
        e_high = random.choice(max_edges)
        u, v = e_high
        
        # Get common neighbors
        common_mask = adj_mask[u] & adj_mask[v]
        if common_mask == 0:
            continue
            
        # Pick a common neighbor to break
        # We want to remove an edge (u, w) or (v, w)
        # Let's pick a random common neighbor w
        w_list = []
        mask = common_mask
        while mask:
            w = (mask & -mask).bit_length() - 1
            w_list.append(w)
            mask &= mask - 1
            
        w = random.choice(w_list)
        
        # Decide whether to remove (u, w) or (v, w)
        # Heuristic: Remove the edge where the other endpoint has higher degree?
        # Or just random.
        if random.random() < 0.5:
            edge_to_remove = (min(u, w), max(u, w))
            remove_u, remove_w = u, w
        else:
            edge_to_remove = (min(v, w), max(v, w))
            remove_u, remove_w = v, w
            
        if edge_to_remove not in edge_set:
            continue
            
        # Remove edge_to_remove
        # This affects books of edges incident to remove_u and remove_w
        
        # 1. Remove from edge_set and adj_mask
        edge_set.remove(edge_to_remove)
        adj_mask[remove_u] &= ~(1 << remove_w)
        adj_mask[remove_w] &= ~(1 << remove_u)
        
        # 2. Update books for affected edges
        # Edges incident to remove_u (excluding remove_w)
        # For each neighbor k of remove_u:
        #    If k is connected to remove_w, then remove_w was a common neighbor of remove_u and k?
        #    Wait.
        #    Book size of (remove_u, k) is |N(remove_u) \cap N(k)|.
        #    We removed edge (remove_u, remove_w). So remove_w is no longer in N(remove_u).
        #    So if remove_w was in N(k), then remove_w was a common neighbor.
        #    So book size decreases by 1.
        
        neighbors_u = []
        mask_u = adj_mask[remove_u]
        while mask_u:
            k = (mask_u & -mask_u).bit_length() - 1
            neighbors_u.append(k)
            mask_u &= mask_u - 1
            
        for k in neighbors_u:
            e_key = (min(remove_u, k), max(remove_u, k))
            if e_key in books:
                if (adj_mask[k] & (1 << remove_w)) != 0:
                    books[e_key] -= 1
                    if books[e_key] < 0: books[e_key] = 0 # Safety
                
        neighbors_w = []
        mask_w = adj_mask[remove_w]
        while mask_w:
            k = (mask_w & -mask_w).bit_length() - 1
            neighbors_w.append(k)
            mask_w &= mask_w - 1
            
        for k in neighbors_w:
            if k == remove_u: continue
            e_key = (min(remove_w, k), max(remove_w, k))
            if e_key in books:
                if (adj_mask[k] & (1 << remove_u)) != 0:
                    books[e_key] -= 1
                    if books[e_key] < 0: books[e_key] = 0

        # Remove the edge's own book entry
        if edge_to_remove in books:
            del books[edge_to_remove]

        # 3. Add a new edge
        # Find a random non-existent edge
        # To be fast, we can pre-generate a list of all possible edges and shuffle?
        # Or just sample.
        # Sampling from 780 edges is fast.
        
        new_edge = None
        attempts = 0
        while attempts < 50:
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b: continue
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != edge_to_remove:
                new_edge = (a, b)
                break
            attempts += 1
            
        if new_edge is None:
            # Revert
            edge_set.add(edge_to_remove)
            adj_mask[remove_u] |= (1 << remove_w)
            adj_mask[remove_w] |= (1 << remove_u)
            # Re-add books? This is complex. 
            # Instead, just skip this iteration if we can't find an edge.
            # But we already modified books. 
            # Better to find new_edge BEFORE modifying books.
            # Let's restructure: Find new_edge first.
            # If not found, continue.
            # But we already modified. 
            # I will just restart the loop iteration if this happens, but I should prevent it.
            # With 401 edges out of 780, probability of finding a new edge in 50 tries is very high.
            # If it fails, we have a broken state. 
            # To be safe, I'll check for new_edge BEFORE removing the old one.
            
            # Since I already removed, I'll just revert and continue.
            # Reverting books is hard. 
            # Let's assume it doesn't happen. 
            # If it does, we might have incorrect books. 
            # I'll add a check: if new_edge is None, revert adj_mask and edge_set, 
            # and recompute books from scratch for safety.
            
            edge_set.add(edge_to_remove)
            adj_mask[remove_u] |= (1 << remove_w)
            adj_mask[remove_w] |= (1 << remove_u)
            books = compute_books_full(adj_mask)
            continue

        a, b = new_edge
        
        # Add new_edge
        edge_set.add(new_edge)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        # Calculate new book size for new_edge
        common_new = (adj_mask[a] & adj_mask[b]).bit_count()
        books[new_edge] = common_new
        
        # Update books for neighbors of a and b
        neighbors_a = []
        mask_a = adj_mask[a]
        while mask_a:
            k = (mask_a & -mask_a).bit_length() - 1
            neighbors_a.append(k)
            mask_a &= mask_a - 1
            
        for k in neighbors_a:
            if k == b: continue
            e_key = (min(a, k), max(a, k))
            if e_key in books:
                if (adj_mask[k] & (1 << b)) != 0:
                    books[e_key] += 1
                    
        neighbors_b = []
        mask_b = adj_mask[b]
        while mask_b:
            k = (mask_b & -mask_b).bit_length() - 1
            neighbors_b.append(k)
            mask_b &= mask_b - 1
            
        for k in neighbors_b:
            if k == a: continue
            e_key = (min(b, k), max(b, k))
            if e_key in books:
                if (adj_mask[k] & (1 << a)) != 0:
                    books[e_key] += 1

        # Update best
        current_max_book = get_max_book(books)
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            best_books = books

    # Final check
    if best_max_book == 7:
        return [list(e) for e in best_edge_set]
        
    # If we didn't reach 7, return the best we found
    return [list(e) for e in best_edge_set]
