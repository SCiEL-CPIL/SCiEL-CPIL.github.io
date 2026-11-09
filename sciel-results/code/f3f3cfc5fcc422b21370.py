import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    n = 40
    target_edges = 401
    
    def run_sa(seed, time_budget):
        random.seed(seed)
        adj = [0] * n
        edges = set()
        books = {}
        
        def add_edge(u, v):
            if u == v:
                return
            if u > v:
                u, v = v, u
            if (u, v) not in edges:
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                books[(u, v)] = (adj[u] & adj[v]).bit_count()
        
        def remove_edge(u, v):
            if u == v:
                return
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                edges.remove((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                del books[(u, v)]
        
        # Init: K20,20 + 1 intra edge
        for u in range(20):
            for v in range(20, 40):
                add_edge(u, v)
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u == v:
            v = (v + 1) % 20
        add_edge(u, v)
        
        while len(edges) < target_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    add_edge(u, v)
        
        # Compute initial stats
        cur_max = max(books.values())
        cur_sum_sq = sum(b * b for b in books.values())
        best_max = cur_max
        best_sum_sq = cur_sum_sq
        best_edges = set(edges)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        T = 2.0
        cooling = 0.99995
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Pick edge to remove
            if random.random() < 0.6:
                # Top 20% by book size
                items = list(books.items())
                items.sort(key=lambda x: -x[1])
                top_k = max(1, len(items) // 5)
                u_rem, v_rem = items[random.randint(0, top_k - 1)][0]
            else:
                u_rem, v_rem = random.choice(list(edges))
            
            # Pick non-edge to add
            while True:
                u_add, v_add = random.choice(all_pairs)
                if (u_add, v_add) not in edges:
                    break
            
            # Affected edges: all edges incident to u_rem, v_rem, u_add, v_add
            affected = set()
            for vtx in (u_rem, v_rem, u_add, v_add):
                mask = adj[vtx]
                while mask:
                    lb = mask & (-mask)
                    w = lb.bit_length() - 1
                    a, b = (vtx, w) if vtx < w else (w, vtx)
                    affected.add((a, b))
                    mask ^= lb
            
            # Save old book values for affected edges
            old_books = {e: books[e] for e in affected if e in books}
            
            # Perform swap
            remove_edge(u_rem, v_rem)
            add_edge(u_add, v_add)
            
            # Update books for affected edges
            for e in affected:
                u, v = e
                if (u, v) in edges:
                    books[(u, v)] = (adj[u] & adj[v]).bit_count()
            
            # Compute new stats
            new_max = 0
            new_sum_sq = 0
            for b in books.values():
                if b > new_max:
                    new_max = b
                new_sum_sq += b * b
            
            # Acceptance
            accepted = False
            if new_max < best_max:
                accepted = True
            elif new_max == best_max and new_sum_sq < best_sum_sq:
                accepted = True
            elif new_max <= best_max:
                delta = new_sum_sq - best_sum_sq
                if T > 0.001:
                    p = math.exp(-max(0, delta) / T)
                    if random.random() < p:
                        accepted = True
            else:
                if T > 0.1 and new_max <= best_max + 1:
                    if random.random() < T * 0.05:
                        accepted = True
            
            if accepted:
                best_max = new_max
                best_sum_sq = new_sum_sq
                best_edges = set(edges)
            else:
                # Rollback
                remove_edge(u_add, v_add)
                add_edge(u_rem, v_rem)
                # Restore books
                for e, val in old_books.items():
                    books[e] = val
                # The added edge (u_add,v_add) was deleted by remove_edge, good
                # The removed edge (u_rem,v_rem) was re-added by add_edge, its book is set
                # But we need to make sure books is consistent
                # After add_edge(u_rem, v_rem), books[(u_rem,v_rem)] is set correctly
                # For other affected edges, we restored from old_books
                # But what about edges that were in affected but not in old_books (i.e., (u_add,v_add) which wasn't in books before)?
                # (u_add,v_add) was not in edges before, so not in books. After rollback, it's not in edges, so not in books. Good.
                # What about edges that are in affected and in edges after rollback but not in old_books?
                # All edges in affected that are in edges after rollback were either in edges before (so in old_books) or are (u_rem,v_rem) which was just added.
                # (u_rem,v_rem) is in old_books (it was in edges before). So it's restored.
                # Actually wait: after add_edge(u_rem, v_rem), books[(u_rem,v_rem)] is set to the new value. But we then restore from old_books which has the old value. That's correct for rollback.
            
            T *= cooling
            if T < 0.0001:
                T = 0.0001
        
        return best_max, best_edges
    
    # Multiple restarts
    best_overall_max = float('inf')
    best_overall_edges = None
    
    # First run gets more time
    runs = [(0, 0.4), (1, 0.3), (2, 0.3)]  # (seed_offset, fraction)
    for seed_off, frac in runs:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 3:
            break
        budget = min(remaining - 1.0, frac * time_limit)
        if budget < 2:
            break
        mx, ed = run_sa(42 + seed_off * 1000, budget)
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall_edges = ed
    
    if best_overall_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_overall_edges]
