import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    def run_sa(seed, time_budget):
        random.seed(seed)
        adj = [0] * n
        edges = set()
        books = {}
        book_counts = [0] * (n + 1)  # Max book size can't exceed n-2
        cur_max = 0
        
        def add_edge(u, v):
            if u == v: return
            if u > v: u, v = v, u
            if (u, v) in edges: return
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            b = (adj[u] & adj[v]).bit_count()
            books[(u, v)] = b
            book_counts[b] += 1
            if b > cur_max:
                cur_max = b
            return b

        def remove_edge(u, v):
            if u == v: return
            if u > v: u, v = v, u
            if (u, v) not in edges: return
            edges.remove((u, v))
            b = books.pop((u, v))
            book_counts[b] -= 1
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

        def get_max_book():
            for b in range(cur_max, -1, -1):
                if book_counts[b] > 0:
                    return b
            return 0

        # Init: Random graph
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:target_edges]:
            add_edge(u, v)
        
        cur_max = get_max_book()
        best_max = cur_max
        best_edges = set(edges)
        best_books = dict(books)
        best_book_counts = book_counts[:]
        
        T = 5.0
        cooling = 0.9999
        end_time = start_time + time_budget
        
        # Pre-compute list of edges for random removal? 
        # Or just iterate over edges set.
        
        while time.time() < end_time:
            # Pick edge to remove
            # Bias towards high book size
            # Simple: random edge
            # Better: sample from top K books
            # For speed, just random edge from list(edges)
            # list(edges) is O(E). E=401. Fast.
            u_rem, v_rem = random.choice(list(edges))
            
            # Pick non-edge to add
            # Random pair
            while True:
                u_add, v_add = random.choice(all_pairs)
                if (u_add, v_add) not in edges:
                    break
            
            # Affected vertices
            verts = {u_rem, v_rem, u_add, v_add}
            
            # Collect affected edges
            affected = set()
            for vtx in verts:
                mask = adj[vtx]
                while mask:
                    lb = mask & (-mask)
                    w = lb.bit_length() - 1
                    a, b = (vtx, w) if vtx < w else (w, vtx)
                    if (a, b) in edges:
                        affected.add((a, b))
                    mask ^= lb
            
            # Save old book sizes
            old_books = {}
            for e in affected:
                old_books[e] = books[e]
            
            # Perform swap
            remove_edge(u_rem, v_rem)
            b_new_add = add_edge(u_add, v_add)
            
            # Update books for affected edges
            for e in affected:
                u, v = e
                if (u, v) in edges:
                    b_new = (adj[u] & adj[v]).bit_count()
                    b_old = old_books[e]
                    if b_new != b_old:
                        book_counts[b_old] -= 1
                        book_counts[b_new] += 1
                        books[e] = b_new
            
            # Update cur_max
            new_max = get_max_book()
            
            # Acceptance
            accepted = False
            if new_max < best_max:
                accepted = True
            elif new_max == best_max:
                # Optional: minimize sum of squares
                # For now, accept to explore
                accepted = True 
            elif new_max <= best_max:
                accepted = True
            else:
                delta = new_max - best_max
                if T > 0.01:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accepted = True
            
            if accepted:
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(edges)
                    best_books = dict(books)
                    best_book_counts = book_counts[:]
                # If new_max == best_max, we don't necessarily update best_edges
                # unless we track sum of squares. 
                # Let's stick to just max for now to keep it simple and fast.
            else:
                # Rollback
                remove_edge(u_add, v_add)
                add_edge(u_rem, v_rem)
                # Restore books
                for e, val in old_books.items():
                    books[e] = val
                    # book_counts were modified in forward pass, need to reverse?
                    # No, we modified book_counts in forward pass.
                    # In rollback, we must restore book_counts to state before forward pass.
                    # This is tricky. 
                    
                # Better approach for rollback:
                # Save book_counts state? No, too expensive.
                # Just recalculate? No.
                
                # Let's re-evaluate the book_counts update.
                # Forward:
                # 1. Remove (u_rem, v_rem). book_counts[b_rem]--
                # 2. Add (u_add, v_add). book_counts[b_add]++
                # 3. Update affected edges.
                
                # Rollback:
                # 1. Remove (u_add, v_add). book_counts[b_add]--
                # 2. Add (u_rem, v_rem). book_counts[b_rem]++
                # 3. Restore affected edges books and book_counts.
                
                # We need to know b_rem and b_add.
                # b_rem = old_books[(u_rem, v_rem)]
                # b_add = b_new_add
                
                # Restore affected edges:
                for e, val in old_books.items():
                    if e == (u_rem, v_rem):
                        continue # Handled by add_edge
                    if e == (u_add, v_add):
                        continue # Handled by remove_edge
                    # e is an edge that existed before and exists after
                    # Its book size changed from val to books[e]
                    b_new = books[e]
                    book_counts[val] += 1
                    book_counts[b_new] -= 1
                    books[e] = val
                
                # Now handle the swap edges
                # We removed (u_add, v_add) in rollback.
                # We added (u_rem, v_rem) in rollback.
                # The book_counts for these were updated in the rollback add/remove calls.
                # Wait, my remove_edge and add_edge update book_counts.
                # So:
                # Forward:
                #   remove_edge(u_rem, v_rem) -> book_counts[b_rem]--
                #   add_edge(u_add, v_add) -> book_counts[b_add]++
                #   update affected -> book_counts changes
                # Rollback:
                #   remove_edge(u_add, v_add) -> book_counts[b_add]--
                #   add_edge(u_rem, v_rem) -> book_counts[b_rem]++
                #   restore affected -> book_counts changes back
                
                # This seems correct.
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_max, best_edges

    best_overall_max = float('inf')
    best_overall_edges = None
    
    # Single long run
    mx, ed = run_sa(42, time_limit - 5)
    if mx < best_overall_max:
        best_overall_max = mx
        best_overall_edges = ed
        
    if best_overall_edges is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for u, v in best_overall_edges]
