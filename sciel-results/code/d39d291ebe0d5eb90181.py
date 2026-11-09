import random
import math

def construct():
    n = 40
    m = 401
    
    # Adjacency as bitsets
    adj = [0] * n
    edge_set = set()
    
    # Book counts: dict mapping (u, v) with u < v to count
    books = {}
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_book(u, v):
        if u > v: u, v = v, u
        return books.get((u, v), 0)

    def add_edge(u, v):
        if u > v: u, v = v, u
        edge_set.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        common_mask = adj[u] & adj[v]
        c = common_mask.bit_count()
        books[(u, v)] = c
        
        cm = common_mask
        w = 0
        while cm:
            if cm & 1:
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                books[e1] = books.get(e1, 0) + 1
                books[e2] = books.get(e2, 0) + 1
            cm >>= 1
            w += 1

    def remove_edge(u, v):
        if u > v: u, v = v, u
        edge_set.discard((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Remove book for (u,v)
        if (u, v) in books:
            del books[(u, v)]
            
        common_mask = adj[u] & adj[v]
        cm = common_mask
        w = 0
        while cm:
            if cm & 1:
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in books:
                    books[e1] -= 1
                    if books[e1] == 0: del books[e1]
                if e2 in books:
                    books[e2] -= 1
                    if books[e2] == 0: del books[e2]
            cm >>= 1
            w += 1

    # Calculate Sum of Squares of Book Sizes
    def get_score():
        s = 0
        for b in books.values():
            s += b * b
        return s

    # Initialize with random graph
    random.shuffle(all_pairs)
    for i in range(m):
        u, v = all_pairs[i]
        add_edge(u, v)
    
    current_score = get_score()
    current_max_book = max(books.values()) if books else 0
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Number of iterations
    # 120s limit. 
    # Each step: pick 2 edges, calc delta (O(deg) ~ 20), update (O(deg)).
    # ~ 50-100 ops per step.
    # 120s / 0.001s/step = 120,000 steps? 
    # Python is slower. Maybe 10,000-50,000 steps.
    # Let's run until time limit or fixed steps.
    
    import time
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick random edge to remove
        if not edge_set:
            break
        u_out, v_out = random.choice(list(edge_set))
        
        # Pick random non-edge to add
        # Efficiently pick non-edge:
        # Total pairs 780. Edges 401. Non-edges 379.
        # Just pick random pair, check if in edge_set.
        while True:
            u_in, v_in = random.choice(all_pairs)
            if (u_in, v_in) not in edge_set:
                break
        
        # Calculate Delta Score for Swap
        # Remove (u_out, v_out)
        # Add (u_in, v_in)
        
        # 1. Score contribution of removing (u_out, v_out)
        # The edge (u_out, v_out) has book B_out.
        # Its removal reduces S by B_out^2.
        # For each common neighbor w of u_out, v_out:
        #   Edge (u_out, w) book decreases by 1: delta = -(2*b+1)
        #   Edge (v_out, w) book decreases by 1: delta = -(2*b+1)
        
        b_out = books.get((min(u_out, v_out), max(u_out, v_out)), 0)
        delta_remove = -b_out * b_out
        
        adj_u_out = adj[u_out]
        adj_v_out = adj[v_out]
        common_out = adj_u_out & adj_v_out
        
        cm = common_out
        w = 0
        while cm:
            if cm & 1:
                e1 = (u_out, w) if u_out < w else (w, u_out)
                e2 = (v_out, w) if v_out < w else (w, v_out)
                b1 = books.get(e1, 0)
                b2 = books.get(e2, 0)
                delta_remove -= (2 * b1 + 1)
                delta_remove -= (2 * b2 + 1)
            cm >>= 1
            w += 1
            
        # 2. Score contribution of adding (u_in, v_in)
        # Note: adj and books are still in state BEFORE swap.
        # When adding (u_in, v_in), the common neighbors are based on CURRENT adj.
        # But wait, if u_in or v_in is connected to u_out or v_out, the common neighbors might change?
        # No, common neighbors of u_in, v_in are $N(u_in) \cap N(v_in)$.
        # The removal of (u_out, v_out) only affects adj[u_out] and adj[v_out].
        # If u_in == u_out, then N(u_in) changes.
        # We must handle cases where the added edge shares vertices with the removed edge.
        
        # Case 1: Disjoint edges {u_out, v_out} and {u_in, v_in} are disjoint sets of vertices.
        # Then N(u_in) and N(v_in) are unaffected by removal.
        # Case 2: They share a vertex.
        
        # To be safe and simple, we can just:
        # 1. Remove edge
        # 2. Calculate score
        # 3. Add edge
        # 4. Calculate score
        # 5. Compare
        # This is 2x the work but much simpler and less bug-prone for the delta calculation.
        # Given the small size, let's try this "Apply and Revert" approach for the evaluation.
        
        # Actually, "Apply and Revert" is O(deg) for remove, O(1) for score calc (if we maintain global S), O(deg) for add.
        # We need to maintain global S.
        
        # Let's restart the inner loop logic with global S maintenance.
        
        # --- Revised Inner Loop ---
        
        # We need to maintain `current_score` (Sum of Squares).
        # We already have `books`.
        
        # Remove (u_out, v_out)
        # Update books and current_score
        b_out = books.get((min(u_out, v_out), max(u_out, v_out)), 0)
        current_score -= b_out * b_out
        if (min(u_out, v_out), max(u_out, v_out)) in books:
            del books[(min(u_out, v_out), max(u_out, v_out))]
            
        adj[u_out] &= ~(1 << v_out)
        adj[v_out] &= ~(1 << u_out)
        edge_set.discard((u_out, v_out))
        
        common_out = adj[u_out] & adj[v_out]
        cm = common_out
        w = 0
        while cm:
            if cm & 1:
                e1 = (u_out, w) if u_out < w else (w, u_out)
                e2 = (v_out, w) if v_out < w else (w, v_out)
                
                b1 = books.get(e1, 0)
                b2 = books.get(e2, 0)
                
                # Update score for e1
                current_score -= b1 * b1
                b1_new = b1 - 1
                current_score += b1_new * b1_new
                if b1_new > 0:
                    books[e1] = b1_new
                else:
                    if e1 in books: del books[e1]
                    
                # Update score for e2
                current_score -= b2 * b2
                b2_new = b2 - 1
                current_score += b2_new * b2_new
                if b2_new > 0:
                    books[e2] = b2_new
                else:
                    if e2 in books: del books[e2]
                    
            cm >>= 1
            w += 1
            
        # Add (u_in, v_in)
        # First, check if they are already connected? No, we picked non-edge.
        # But we must update adj first to get common neighbors?
        # No, common neighbors for the NEW edge (u_in, v_in) are $N(u_in) \cap N(v_in)$ in the CURRENT state (after removal).
        
        adj[u_in] |= (1 << v_in)
        adj[v_in] |= (1 << u_in)
        edge_set.add((u_in, v_in))
        
        common_in = adj[u_in] & adj[v_in]
        c_in = common_in.bit_count()
        
        # Update score for new edge (u_in, v_in)
        e_new = (min(u_in, v_in), max(u_in, v_in))
        current_score += c_in * c_in
        books[e_new] = c_in
        
        cm = common_in
        w = 0
        while cm:
            if cm & 1:
                e1 = (u_in, w) if u_in < w else (w, u_in)
                e2 = (v_in, w) if v_in < w else (w, v_in)
                
                b1 = books.get(e1, 0)
                b2 = books.get(e2, 0)
                
                # Update score for e1
                current_score -= b1 * b1
                b1_new = b1 + 1
                current_score += b1_new * b1_new
                books[e1] = b1_new
                
                # Update score for e2
                current_score -= b2 * b2
                b2_new = b2 + 1
                current_score += b2_new * b2_new
                books[e2] = b2_new
                
            cm >>= 1
            w += 1
            
        # Now state is swapped.
        # Calculate new max book?
        # We can just track if we improved the max book or the score.
        # Since calculating max book is O(M), we can do it periodically or just rely on Score.
        # Minimizing Sum of Squares is a good proxy for minimizing Max.
        
        # Acceptance
        # We need to know if this is an improvement.
        # We don't have the old score stored separately? 
        # We modified current_score in place.
        # We need to compare with the score BEFORE the swap.
        
        # Let's restructure: 
        # 1. Save old score.
        # 2. Perform swap (update books, adj, current_score).
        # 3. If current_score < old_score OR (random < exp(delta/T)):
        #       Accept (do nothing, state is already swapped)
        #    Else:
        #       Revert swap (update books, adj, current_score back to old)
        
        # This is messy. Better:
        # 1. Calculate Delta S analytically.
        # This is what I tried first. Let's go back to that, but handle the shared vertex cases carefully or just assume random edges are likely disjoint?
        # With 40 vertices, probability of disjoint is high.
        # P(shared) = 1 - P(disjoint).
        # P(disjoint) = (38*37)/(40*39) approx 0.9.
        # So 10% of the time we have shared vertices.
        # If we just use the analytical delta for disjoint, and for shared, we can fall back to "Apply, Check, Revert" or just reject shared moves initially?
        # Rejecting shared moves reduces connectivity. Bad.
        
        # Let's use the "Apply, Check, Revert" method but optimize the "Check".
        # We maintain `current_score`.
        # We save `old_score = current_score`.
        # We perform the swap updates (which update `current_score`).
        # If `current_score < old_score`: Accept.
        # Else: 
        #    prob = exp((old_score - current_score) / T) # Note: delta = new - old. If new > old, delta > 0. We want exp(-delta/T).
        #    if random() < prob: Accept.
        #    Else: Revert.
        
        # Reverting is the same as swapping back.
        # So we need a function `swap(u1, v1, u2, v2)` that updates state and score.
        # And we call it twice if we revert.
        
        # This doubles the work. 
        # Alternative: Store the changes made and undo them?
        # The changes are:
        # - Edge (u_out, v_out) removed.
        # - Edge (u_in, v_in) added.
        # - Books of neighbors updated.
        
        # Given the complexity, let's just run a simple Hill Climbing on Max Book directly, but faster.
        # Or, stick to the current program's approach but fix the local search to be deterministic and fast.
        
        # Let's look at the current program again.
        # It does Greedy, then Random Swaps.
        # The Random Swaps are very slow because they do `max(books.values())` every time.
        
        # Improvement:
        # Maintain `max_book` and `count_max`.
        # When updating books, update these stats.
        
        # Let's implement the "Maintain Stats" approach with Random Swaps.
        
    # --- Final Decision ---
    # I will rewrite the program to use Simulated Annealing on Sum of Squares, 
    # with an efficient "Apply and Revert" mechanism.
    # To avoid the complexity of reverting, I will calculate the delta analytically 
    # for the general case, and if vertices overlap, I will just skip the move 
    # or handle it with a slower path. 
    # Actually, for n=40, "Apply and Revert" is fast enough if we just do it.
    # 2 * O(deg) per step. deg ~ 20. 40 ops. 
    # 100,000 steps = 4,000,000 ops. Python can do ~10-50M ops/sec. 
    # So 100,000 steps should take < 1s. 
    # We can run for 110s easily.
    
    # Let's implement the "Apply and Revert" SA on Sum of Squares.
    
    # Reset state for the real run
    edge_set.clear()
    books.clear()
    adj = [0] * n
    for i in range(m):
        u, v = all_pairs[i]
        add_edge(u, v)
        
    current_score = get_score()
    
    T = 5.0
    cooling = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick random edge to remove
        u_out, v_out = random.choice(list(edge_set))
        
        # Pick random non-edge to add
        while True:
            u_in, v_in = random.choice(all_pairs)
            if (u_in, v_in) not in edge_set:
                break
                
        # Save old score
        old_score = current_score
        
        # Perform Swap: Remove (u_out, v_out), Add (u_in, v_in)
        
        # 1. Remove (u_out, v_out)
        key_out = (min(u_out, v_out), max(u_out, v_out))
        b_out = books.get(key_out, 0)
        current_score -= b_out * b_out
        if key_out in books: del books[key_out]
        
        adj[u_out] &= ~(1 << v_out)
        adj[v_out] &= ~(1 << u_out)
        edge_set.discard(key_out)
        
        common_out = adj[u_out] & adj[v_out]
        cm = common_out
        w = 0
        while cm:
            if cm & 1:
                e1 = (u_out, w) if u_out < w else (w, u_out)
                e2 = (v_out, w) if v_out < w else (w, v_out)
                
                b1 = books.get(e1, 0)
                b2 = books.get(e2, 0)
                
                current_score -= b1 * b1
                b1_new = b1 - 1
                current_score += b1_new * b1_new
                if b1_new > 0: books[e1] = b1_new
                elif e1 in books: del books[e1]
                
                current_score -= b2 * b2
                b2_new = b2 - 1
                current_score += b2_new * b2_new
                if b2_new > 0: books[e2] = b2_new
                elif e2 in books: del books[e2]
                
            cm >>= 1
            w += 1
            
        # 2. Add (u_in, v_in)
        adj[u_in] |= (1 << v_in)
        adj[v_in] |= (1 << u_in)
        edge_set.add((u_in, v_in))
        
        common_in = adj[u_in] & adj[v_in]
        c_in = common_in.bit_count()
        
        key_in = (min(u_in, v_in), max(u_in, v_in))
        current_score += c_in * c_in
        books[key_in] = c_in
        
        cm = common_in
        w = 0
        while cm:
            if cm & 1:
                e1 = (u_in, w) if u_in < w else (w, u_in)
                e2 = (v_in, w) if v_in < w else (w, v_in)
                
                b1 = books.get(e1, 0)
                b2 = books.get(e2, 0)
                
                current_score -= b1 * b1
                b1_new = b1 + 1
                current_score += b1_new * b1_new
                books[e1] = b1_new
                
                current_score -= b2 * b2
                b2_new = b2 + 1
                current_score += b2_new * b2_new
                books[e2] = b2_new
                
            cm >>= 1
            w += 1
            
        # 3. Acceptance
        delta = current_score - old_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            pass
        else:
            # Revert: Swap back
            # Remove (u_in, v_in)
            key_in = (min(u_in, v_in), max(u_in, v_in))
            b_in = books.get(key_in, 0)
            current_score -= b_in * b_in
            if key_in in books: del books[key_in]
            
            adj[u_in] &= ~(1 << v_in)
            adj[v_in] &= ~(1 << u_in)
            edge_set.discard(key_in)
            
            common_in = adj[u_in] & adj[v_in]
            cm = common_in
            w = 0
            while cm:
                if cm & 1:
                    e1 = (u_in, w) if u_in < w else (w, u_in)
                    e2 = (v_in, w) if v_in < w else (w, v_in)
                    
                    b1 = books.get(e1, 0)
                    b2 = books.get(e2, 0)
                    
                    current_score -= b1 * b1
                    b1_new = b1 - 1
                    current_score += b1_new * b1_new
                    if b1_new > 0: books[e1] = b1_new
                    elif e1 in books: del books[e1]
                    
                    current_score -= b2 * b2
                    b2_new = b2 - 1
                    current_score += b2_new * b2_new
                    if b2_new > 0: books[e2] = b2_new
                    elif e2 in books: del books[e2]
                    
                cm >>= 1
                w += 1
                
            # Add (u_out, v_out)
            adj[u_out] |= (1 << v_out)
            adj[v_out] |= (1 << u_out)
            edge_set.add(key_out)
            
            common_out = adj[u_out] & adj[v_out]
            c_out = common_out.bit_count()
            
            current_score += c_out * c_out
            books[key_out] = c_out
            
            cm = common_out
            w = 0
            while cm:
                if cm & 1:
                    e1 = (u_out, w) if u_out < w else (w, u_out)
                    e2 = (v_out, w) if v_out < w else (w, v_out)
                    
                    b1 = books.get(e1, 0)
                    b2 = books.get(e2, 0)
                    
                    current_score -= b1 * b1
                    b1_new = b1 + 1
                    current_score += b1_new * b1_new
                    books[e1] = b1_new
                    
                    current_score -= b2 * b2
                    b2_new = b2 + 1
                    current_score += b2_new * b2_new
                    books[e2] = b2_new
                    
                cm >>= 1
                w += 1
                
        T *= cooling
        
    return [[u, v] for u, v in edge_set]
