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

    def balanced_bipartite_init(n, m):
        adj_mask = [0] * n
        edge_set = set()
        set_a = list(range(20))
        set_b = list(range(20, 40))
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        if random.random() < 0.5:
            u, v = random.sample(set_a, 2)
        else:
            u, v = random.sample(set_b, 2)
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        return edge_set, adj_mask

    def random_regular_init(n, m):
        adj_mask = [0] * n
        edge_set = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:m]:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        degrees = [adj_mask[i].bit_count() for i in range(n)]
        for _ in range(200):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            moved = False
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            random.shuffle(neigh_max)
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edge_set.discard((min(max_idx, u), max(max_idx, u)))
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edge_set.add((min(min_idx, u), max(min_idx, u)))
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    moved = True
                    break
            if not moved:
                break
        return edge_set, adj_mask

    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    init_methods = [balanced_bipartite_init, random_regular_init]
    
    for init_method in init_methods:
        edge_set, adj_mask = init_method(n, m)
        books = compute_books(adj_mask)
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
            best_max_book = current_max_book
            best_sum_sq = current_sum_sq
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
            
    edge_set = best_edge_set
    adj_mask = best_adj_mask
    books = compute_books(adj_mask)
    current_max_book = best_max_book
    current_sum_sq = best_sum_sq
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    if current_max_book == 7:
        return [list(e) for e in edge_set]
        
    T = 2.0
    T_min = 0.05
    cooling = 0.9995
    
    # Precompute neighbors for fast access
    # We will update this dynamically
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Use a list for fast random choice
        # We maintain edge_list in sync with edge_set
        # For now, use list(edge_set) but optimize later if needed
        
        # Optimization: Keep a list of edges
        # This is getting complex. Let's just speed up the inner loop.
        
        # Pick e_remove
        if random.random() < 0.8:
            # Top 20% by book size
            # Sorting 401 items is fast
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 5)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        u, v = min(u, v), max(u, v)
        
        # Pick e_add
        candidates = []
        for _ in range(30):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                candidates.append((a, b))
                
        if not candidates:
            T *= cooling
            continue
            
        best_cand = None
        best_common = 100
        for a, b in candidates:
            common = (adj_mask[a] & adj_mask[b]).bit_count()
            if common < best_common:
                best_common = common
                best_cand = (a, b)
                
        if best_cand is None:
            T *= cooling
            continue
            
        a, b = best_cand
        
        # Identify affected edges
        # Edges incident to u, v, a, b
        affected_edges = set()
        for x in [u, v, a, b]:
            mask = adj_mask[x]
            while mask:
                y = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if x < y:
                    affected_edges.add((x, y))
                else:
                    affected_edges.add((y, x))
                    
        # Save old values
        old_vals = {}
        for e in affected_edges:
            if e in books:
                old_vals[e] = books[e]
                
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add((a, b))
        
        # Update books for affected edges
        # Note: The set of affected edges might change slightly if (a,b) was not incident to them?
        # No, (a,b) is incident to a and b.
        # So the set of edges incident to u,v,a,b is the correct set to update.
        # However, if (a,b) was already in the graph? No, we checked.
        # If (u,v) was removed, it's no longer in edge_set.
        
        # Recompute books for affected edges
        # We need to ensure we update all edges that might have changed.
        # An edge (x,y) changes if x or y is in {u,v,a,b}.
        # So we iterate over neighbors of u,v,a,b in the NEW graph?
        # Or just recompute for the union of old and new affected edges.
        
        # Let's just recompute for all edges incident to u,v,a,b in the new state.
        new_affected = set()
        for x in [u, v, a, b]:
            mask = adj_mask[x]
            while mask:
                y = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if x < y:
                    new_affected.add((x, y))
                else:
                    new_affected.add((y, x))
                    
        # Union of old and new affected edges to be safe
        all_affected = affected_edges.union(new_affected)
        
        for e in all_affected:
            if e in edge_set:
                x, y = e
                books[e] = (adj_mask[x] & adj_mask[y]).bit_count()
            else:
                if e in books:
                    del books[e]
                    
        # Update max and sum_sq
        # We can just recompute max and sum_sq from the books dict?
        # 401 items. Fast.
        new_max_book = get_max_book(books)
        new_sum_sq = get_sum_sq_books(books)
        
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 500)):
                    accept = True
        else:
            if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                if best_max_book == 7:
                    return [list(e) for e in best_edge_set]
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            edge_set.remove((a, b))
            edge_set.add(e_remove)
            # Restore books
            for e in all_affected:
                if e in old_vals:
                    books[e] = old_vals[e]
                else:
                    # If it was added and now removed?
                    # If e is in edge_set (reverted), it should be in books.
                    # If e was not in old_vals, it means it wasn't in books before?
                    # This logic is tricky.
                    # Better to just recompute books for affected edges after revert?
                    # Or save the entire books dict? No, too slow.
                    # Let's just recompute for all_affected.
                    pass
                    
            # Safest revert: recompute books for all_affected
            for e in all_affected:
                if e in edge_set:
                    x, y = e
                    books[e] = (adj_mask[x] & adj_mask[y]).bit_count()
                else:
                    if e in books:
                        del books[e]
                        
        T *= cooling
        
    return [list(e) for e in best_edge_set]
