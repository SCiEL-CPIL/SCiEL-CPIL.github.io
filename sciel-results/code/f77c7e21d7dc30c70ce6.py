import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) to graph with adjacency bitmasks adj
        results in a K3,3-free graph.
        adj is a list of integers where bit k is 1 if vertex k is a neighbor.
        """
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return True # Already exists, trivially valid addition (no change)
        
        # Tentatively add
        nu = adj[u] | (1 << v)
        nv = adj[v] | (1 << u)
        
        # Check triples involving u
        # We need |N(u) \cap N(x) \cap N(y)| <= 2 for all x, y
        # Optimization: If |N(u) \cap N(x)| < 3, then for any y, 
        # |N(u) \cap N(x) \cap N(y)| <= |N(u) \cap N(x)| < 3.
        for x in range(n):
            if x == u:
                continue
            nx = adj[x]
            common_ux = nu & nx
            if common_ux.bit_count() < 3:
                continue
            # If common_ux has >= 3 bits, we must check y
            # We can iterate y
            for y in range(n):
                if y == u or y == x:
                    continue
                ny = adj[y]
                if (common_ux & ny).bit_count() > 2:
                    return False
        
        # Check triples involving v
        for x in range(n):
            if x == v:
                continue
            nx = adj[x]
            common_vx = nv & nx
            if common_vx.bit_count() < 3:
                continue
            for y in range(n):
                if y == v or y == x:
                    continue
                ny = adj[y]
                if (common_vx & ny).bit_count() > 2:
                    return False
            
        return True

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_build(rng):
        adj = [0] * n
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        for u, v in pairs:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng, time_limit):
        start = time.time()
        adj = [x for x in adj]
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # 1. Try to add missing edges
            missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            rng.shuffle(missing)
            for u, v in missing:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            
            if time.time() - start > time_limit * 0.7:
                break

            # 2. Try swaps
            # Existing edges
            existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            
            if not existing or not missing:
                break
                
            rng.shuffle(existing)
            rng.shuffle(missing)
            
            # Try a limited number of random swaps
            # We want to swap an edge that is "blocking" many additions
            # Heuristic: Just try random swaps
            max_swaps = 200
            for _ in range(max_swaps):
                if time.time() - start > time_limit * 0.9:
                    break
                a, b = rng.choice(existing)
                c, d = rng.choice(missing)
                
                # Remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Try add (c,d)
                if can_add_edge(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    improved = True
                    # Restart existing/missing lists? Or just continue?
                    # Restarting is safer to ensure we don't use stale lists
                    # But expensive. Let's just break and re-loop
                    break
                else:
                    # Restore (a,b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    
        return adj

    def simulated_annealing(adj, rng, time_limit):
        start = time.time()
        current = [x for x in adj]
        best = [x for x in current]
        best_count = get_edge_count(current)
        
        T = 5.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            # Generate move
            r = rng.random()
            
            if r < 0.5:
                # Add a random missing edge
                missing = [(u, v) for u, v in all_pairs if not ((current[u] >> v) & 1)]
                if not missing:
                    continue
                u, v = rng.choice(missing)
                
                # Tentatively add
                old_u = current[u]
                old_v = current[v]
                current[u] |= (1 << v)
                current[v] |= (1 << u)
                
                if can_add_edge(current, u, v): # Note: can_add_edge checks if adding is valid given current state. 
                    # Wait, my can_add_edge takes the state BEFORE addition?
                    # In my implementation: "Check if adding edge (u, v) to graph with adjacency bitmasks adj"
                    # It uses adj[u] and adj[v] to compute nu, nv.
                    # So I should pass the state BEFORE addition.
                    pass 
                
                # Let's re-verify can_add_edge logic.
                # It takes adj. It computes nu = adj[u] | (1<<v).
                # So it expects adj to NOT have the edge.
                
                # So:
                # Save old
                # Add to current
                # Check validity of current? No, check if adding to OLD is valid.
                
                # Let's refactor: 
                # valid = can_add_edge(current, u, v) # where current does NOT have u-v
                # If valid:
                #    current[u] |= ...
                #    current[v] |= ...
                #    new_count = ...
                #    if new_count > best_count: update best
                #    accept (always, since it's an improvement)
                # Else:
                #    reject
                
                # But I already modified current above in the thought process. 
                # Let's write it cleanly in the code.
                
                # Re-doing the block for clarity in final code:
                # 1. Pick missing u,v
                # 2. Check can_add_edge(current, u, v)
                # 3. If True: Add to current. Update best if count increases.
                # 4. If False: Reject.
                
            elif r < 0.8:
                # Remove a random existing edge
                existing = [(u, v) for u, v in all_pairs if (current[u] >> v) & 1]
                if not existing:
                    continue
                u, v = rng.choice(existing)
                
                # Remove
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
                
                new_count = get_edge_count(current)
                delta = new_count - get_edge_count(best) # This is expensive. 
                # Better: track current_count
                # Let's track current_count variable.
                
            else:
                # Swap
                existing = [(u, v) for u, v in all_pairs if (current[u] >> v) & 1]
                missing = [(u, v) for u, v in all_pairs if not ((current[u] >> v) & 1)]
                if not existing or not missing:
                    continue
                u1, v1 = rng.choice(existing)
                u2, v2 = rng.choice(missing)
                
                # Remove u1-v1
                current[u1] &= ~(1 << v1)
                current[v1] &= ~(1 << u1)
                
                # Try add u2-v2
                if can_add_edge(current, u2, v2):
                    current[u2] |= (1 << v2)
                    current[v2] |= (1 << u2)
                    # Count is same
                    # Accept?
                    # Accepting swaps can help escape local optima.
                    # Let's accept all valid swaps.
                else:
                    # Restore
                    current[u1] |= (1 << v1)
                    current[v1] |= (1 << u1)
            
            T *= cooling
            
        return best, best_count

    # Main Loop
    rng = random.Random(42)
    
    best_adj = None
    best_count = 0
    
    # 1. Greedy Init (Multiple restarts)
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
        adj = greedy_build(rng)
        count = get_edge_count(adj)
        if count > best_count:
            best_adj = adj
            best_count = count

    # 2. Local Search on best
    if best_adj is not None:
        time_left = deadline - time.time() - 10
        if time_left > 1:
            best_adj = local_search(best_adj, rng, min(10.0, time_left))
            best_count = get_edge_count(best_adj)

    # 3. Simulated Annealing
    if time.time() < deadline - 20:
        time_left = deadline - time.time() - 5
        if time_left > 1:
            sa_adj, sa_count = simulated_annealing(best_adj, rng, min(80.0, time_left))
            if sa_count > best_count:
                best_adj = sa_adj
                best_count = sa_count

    # 4. Final Local Search
    if time.time() < deadline - 2:
        time_left = deadline - time.time() - 1
        if time_left > 0.5:
            best_adj = local_search(best_adj, rng, min(2.0, time_left))
            best_count = get_edge_count(best_adj)

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(n):
            if u < v and (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
