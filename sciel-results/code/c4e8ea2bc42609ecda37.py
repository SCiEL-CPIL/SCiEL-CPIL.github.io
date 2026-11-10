import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 20
    all_mask = (1 << n) - 1

    def common_count(adj, i, j, k):
        return (adj[i] & adj[j] & adj[k]).bit_count()

    def is_valid(adj):
        # No triple of vertices has >=3 common neighbors
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def check_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        After adding, u and v become neighbors. New triples involving u or v
        where the new edge creates a 3rd common neighbor."""
        # The new edge makes u adjacent to v and v adjacent to u.
        # For any pair (a,b) not involving u or v: common neighbors unchanged.
        # Triples including u: (u, a, b) -> common neighbors = N(u) & N(a) & N(b)
        #   But N(u) gains v. So new common count = old + (1 if v in N(a) and v in N(b) else 0)
        #   We need new count < 3, i.e., old count + indicator < 3
        #   If old count was 2 and v is common neighbor of a and b, then new = 3 -> violation
        # Similarly for triples including v.
        # Also triple (u, v, a): common neighbors = N(u)&N(v)&N(a) after addition
        #   N(u) now includes v, N(v) now includes u.
        #   N(u)&N(v) = (old_N(u) ∪ {v}) & (old_N(v) ∪ {u})
        #   = (old_N(u) & old_N(v)) ∪ (old_N(u) ∩ {u}) ∪ (old_N(v) ∩ {v}) ∪ ({u}∩{v})
        #   Since u not in old_N(u) and v not in old_N(v) (simple graph, no self-loops):
        #   = old_N(u) & old_N(v)
        #   Wait, that's not right. Let me reconsider.
        #   After adding edge u-v: N(u) = old_N(u) ∪ {v}, N(v) = old_N(v) ∪ {u}
        #   N(u) & N(v) = (old_N(u) ∪ {v}) & (old_N(v) ∪ {u})
        #   = (old_N(u) & old_N(v)) | (old_N(u) & {u}) | ({v} & old_N(v)) | ({v} & {u})
        #   old_N(u) & {u} = 0 (no self-loop), {v} & old_N(v) = 0, {v}&{u}=0
        #   So N(u) & N(v) = old_N(u) & old_N(v)
        #   Thus for triple (u,v,a): common = (old_N(u) & old_N(v) & N(a))
        #   This is the same as before adding the edge! So no new violation from (u,v,a).
        
        # So only concern: triples (u,a,b) and (v,a,b) where a,b != u,v
        # For (u,a,b): new common = (old_N(u) ∪ {v}) & N(a) & N(b)
        #   = (old_N(u) & N(a) & N(b)) ∪ ({v} & N(a) & N(b))
        #   count = old_count(u,a,b) + (1 if v in N(a) and v in N(b) else 0)
        #   Need this < 3.
        
        nu = adj[u]
        nv = adj[v]
        
        # Check triples (u, a, b) for all a < b, a,b != u
        for a in range(n):
            if a == u:
                continue
            na = adj[a]
            for b in range(a + 1, n):
                if b == u:
                    continue
                nb = adj[b]
                old_cnt = (nu & na & nb).bit_count()
                if old_cnt >= 2:
                    # If v is also common neighbor of a and b, new count = old+1 >= 3
                    if (nv & na & nb).bit_count() >= 1:
                        return False
        # Check triples (v, a, b) for all a < b, a,b != v
        for a in range(n):
            if a == v:
                continue
            na = adj[a]
            for b in range(a + 1, n):
                if b == v:
                    continue
                nb = adj[b]
                old_cnt = (nv & na & nb).bit_count()
                if old_cnt >= 2:
                    if (nu & na & nb).bit_count() >= 1:
                        return False
        return True

    def check_remove(adj, u, v):
        """After removing edge (u,v), check if graph is still K3,3-free.
        Removing an edge can only decrease common neighbor counts, so it's always safe."""
        return True

    def try_add(adj, u, v):
        if adj[u] & (1 << v):
            return False
        if not check_add(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def try_remove(adj, u, v):
        if not (adj[u] & (1 << v)):
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    # Phase 1: Greedy construction
    def greedy_build():
        adj = [0] * n
        # Random vertex ordering
        order = list(range(n))
        random.shuffle(order)
        for u in order:
            # Try to add edges from u to all other vertices in random order
            candidates = [v for v in range(n) if v != u]
            random.shuffle(candidates)
            for v in candidates:
                try_add(adj, u, v)
        return adj

    # Phase 2: Local search with swaps
    def local_search(adj, time_limit):
        best_adj = list(adj)
        best_count = edge_count(best_adj)
        cur_adj = list(adj)
        cur_count = best_count
        
        t_end = time.time() + time_limit
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < t_end:
            if no_improve > max_no_improve:
                break
            
            # Pick random u, v
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            
            has_uv = bool(cur_adj[u] & (1 << v))
            
            if has_uv:
                # Try removing (u,v) and adding (x,y)
                # Or just removing
                # Try swap: remove (u,v), add (x,y)
                x = random.randrange(n)
                y = random.randrange(n)
                if x == y or x == u or x == v or y == u or y == v:
                    continue
                if cur_adj[x] & (1 << y):
                    continue
                # Check if removing (u,v) and adding (x,y) is valid
                # First remove (u,v)
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                # Now try adding (x,y)
                if check_add(cur_adj, x, y):
                    cur_adj[x] |= (1 << y)
                    cur_adj[y] |= (1 << x)
                    new_count = cur_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = list(cur_adj)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    no_improve += 1
            else:
                # Try adding (u,v)
                if check_add(cur_adj, u, v):
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(cur_adj)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        return best_adj, best_count

    # Phase 3: Simulated annealing
    def simulated_annealing(adj, time_limit):
        cur_adj = list(adj)
        cur_count = edge_count(cur_adj)
        best_adj = list(cur_adj)
        best_count = cur_count
        
        t_start = time.time()
        t_end = t_start + time_limit
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < t_end:
            T = max(T * cooling, T_min)
            
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            
            has_uv = bool(cur_adj[u] & (1 << v))
            
            if has_uv:
                # Remove (u,v) - always valid
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                new_count = cur_count - 1
                delta = new_count - cur_count
                # Accept with probability exp(delta/T)
                if delta >= 0 or random.random() < (1.0 / (T + 1e-10)):
                    cur_count = new_count
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(cur_adj)
                else:
                    # Revert
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
            else:
                # Try adding (u,v)
                if check_add(cur_adj, u, v):
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    new_count = cur_count + 1
                    delta = new_count - cur_count
                    if delta >= 0:
                        cur_count = new_count
                        if cur_count > best_count:
                            best_count = cur_count
                            best_adj = list(cur_adj)
                    # delta is always +1 here, so always accept
                # else: can't add, skip
            
        return best_adj, best_count

    # Main search loop
    best_adj = None
    best_count = -1
    
    # Initial greedy builds
    for i in range(5):
        if time.time() > deadline:
            break
        adj = greedy_build()
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
    
    # Local search from best
    if best_adj is not None and time.time() < deadline:
        time_left = deadline - time.time()
        best_adj, best_count = local_search(best_adj, min(time_left * 0.4, 30))
    
    # More greedy + local search iterations
    while time.time() < deadline - 10:
        adj = greedy_build()
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
        
        time_left = deadline - time.time()
        if time_left > 5:
            ls_time = min(time_left * 0.3, 15)
            adj2, cnt2 = local_search(adj, ls_time)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = list(adj2)
    
    # Final simulated annealing
    if best_adj is not None and time.time() < deadline - 5:
        sa_time = deadline - time.time() - 2
        if sa_time > 3:
            adj3, cnt3 = simulated_annealing(best_adj, sa_time)
            if cnt3 > best_count:
                best_count = cnt3
                best_adj = list(adj3)
    
    # Verify and convert to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Final verification
    if not is_valid(best_adj):
        # Fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
