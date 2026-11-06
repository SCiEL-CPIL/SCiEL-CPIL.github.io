import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 140  # Leave margin for 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)  # 190
    
    # Helper to check if adding edge (u,v) to adj is valid (no K2,3)
    # K2,3 exists if there are two vertices with 3+ common neighbors
    # Adding edge (u,v) can create K2,3 if:
    # 1. u and v already have 3+ common neighbors (direct)
    # 2. There exists w such that w is common neighbor of u and some x, and v is common neighbor of x and some y
    #    Actually simpler: after adding (u,v), check all pairs for 3+ common neighbors
    # Most efficient: just check if any pair would have 3+ common neighbors
    # The critical check: for any w, if w is adjacent to both u and v, then we need
    # |N(w) ∩ N(u) ∩ {v}| + |N(w) ∩ N(v) ∩ {u}| ... 
    # Simplest correct approach: after adding, no pair has 3+ common neighbors
    # Before adding, all pairs are fine. After adding (u,v):
    # - (u,v) pair: common neighbors = N(u) ∩ N(v) - must be <= 2
    # - For any w != u,v: N(w) gains {u} if w~v, gains {v} if w~u
    #   So N(w) ∩ N(x) changes only if x is u or v
    #   Pairs to check: (u,v), (w,u) for w~v, (w,v) for w~u
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        # Check (u,v): common neighbors must be <= 2
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # For each w adjacent to v: check pair (w, u)
        # N(w) ∩ N(u) currently has some count c. After adding (u,v), if w~v, 
        # N(w) gains v (if not already), N(u) gains v. So N(w) ∩ N(u) gains v if v~w and v~u
        # Wait: we're adding (u,v). So v becomes neighbor of u.
        # For pair (w, u): common neighbors = N(w) ∩ N(u)
        # Before: N(w) ∩ N(u) 
        # After: N(w) ∩ (N(u) ∪ {v}) = (N(w) ∩ N(u)) ∪ (N(w) ∩ {v})
        # So new count = old_count + (1 if v in N(w) else 0) - (1 if v in N(w) ∩ N(u) else 0)
        # If v in N(w): new_count = old_count + 1 - (1 if v in N(u) else 0)
        # Since we're adding (u,v), v not in N(u) yet. So if v in N(w), new_count = old_count + 1
        # Need old_count + 1 <= 2, i.e., old_count <= 1
        mask = adj[v]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if w == u:
                mask ^= lb
                continue
            if (adj[w] & adj[u]).bit_count() > 1:
                return False
            mask ^= lb
        
        # For each w adjacent to u: check pair (w, v)
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if w == v:
                mask ^= lb
                continue
            if (adj[w] & adj[v]).bit_count() > 1:
                return False
            mask ^= lb
        
        return True
    
    # Check entire graph for K2,3
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue
                # If u,v not adjacent, check if they'd form K2,3
                # Actually we need to check all pairs
        # Full check: for every pair (u,v), count common neighbors
        # If >= 3, invalid
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
    
    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lb
        return edges
    
    # Greedy construction with random ordering
    def greedy():
        adj = [0] * n
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        count = 0
        for u, v in edge_list:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        return adj, count
    
    # Greedy with degree balancing: prefer edges that balance degrees
    def greedy_balanced():
        adj = [0] * n
        degrees = [0] * n
        remaining = set(all_edges)
        count = 0
        while remaining:
            if time.time() > deadline - 15:
                break
            # Pick edge that minimizes max degree after addition
            best_edge = None
            best_max_deg = float('inf')
            # Sample for speed
            sample_size = min(len(remaining), 30)
            sample = random.sample(list(remaining), sample_size)
            for u, v in sample:
                if not can_add(adj, u, v):
                    continue
                new_max = max(degrees[u] + 1, degrees[v] + 1)
                if new_max < best_max_deg:
                    best_max_deg = new_max
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            count += 1
            remaining.discard(best_edge)
        return adj, count
    
    # Local search: try adding missing edges
    def local_add(adj, count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    improved = True
        return adj, count
    
    # Edge swap: remove one, try to add up to two
    def local_swap(adj, count, max_iters=200):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count == 0:
                break
            
            # Pick random edge to remove
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two random edges (prefer edges involving u or v)
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            # Move edges involving u or v to front
            edge_list.sort(key=lambda e: 0 if (e[0] == u or e[1] == u or e[0] == v or e[1] == v) else 1)
            random.shuffle([e for e in edge_list if e[0] == u or e[1] == u or e[0] == v or e[1] == v])
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # 2-opt style: remove two edges, try to add three
    def local_swap2(adj, count, max_iters=100):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count < 2:
                break
            
            # Pick two random edges to remove
            e1 = random.choice(all_edges)
            e2 = random.choice(all_edges)
            if e1 == e2:
                continue
            if not (adj[e1[0]] & (1 << e1[1])):
                continue
            if not (adj[e2[0]] & (1 << e2[1])):
                continue
            
            # Remove both
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            count -= 2
            
            # Try to add three edges, prioritizing edges near removed ones
            added = 0
            involved = {e1[0], e1[1], e2[0], e2[1]}
            candidates = [(a, b) for a, b in all_edges if a in involved or b in involved]
            random.shuffle(candidates)
            for a, b in candidates:
                if added >= 3:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            if added < 3:
                edge_list = all_edges[:]
                random.shuffle(edge_list)
                for a, b in edge_list:
                    if added >= 3:
                        break
                    if adj[a] & (1 << b):
                        continue
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        count += 1
                        added += 1
            
            if added < 1:
                # Restore both
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                count += 2
        
        return adj, count
    
    # Degree-based greedy: prioritize adding edges between low-degree vertices
    def degree_greedy():
        adj = [0] * n
        degrees = [0] * n
        edge_list = all_edges[:]
        # Sort by degree sum (lower first) - but degrees change, so use current
        count = 0
        # Use a priority approach: repeatedly pick best edge
        remaining = set(all_edges)
        while remaining:
            best_edge = None
            best_score = float('inf')
            # Sample a subset for speed
            sample_size = min(len(remaining), 50)
            sample = random.sample(list(remaining), sample_size)
            for u, v in sample:
                score = degrees[u] + degrees[v]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                count += 1
            remaining.discard(best_edge)
        return adj, count
    
    # Simulated annealing-like: accept worse moves with probability
    def simulated_annealing(adj, count, steps=5000):
        T = 1.0
        T_min = 0.01
        for i in range(steps):
            if time.time() > deadline - 2:
                break
            T = T_min + (T - T_min) * (1 - i / steps)
            
            # Pick random edge swap
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two
            added = 0
            edge_list = random.sample(all_edges, min(50, num_edges_total))
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Many diverse greedy starts
    start_time = time.time()
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        if i % 2 == 0:
            adj, count = greedy()
        else:
            adj, count = greedy_balanced()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Degree-based greedy
    for i in range(10):
        if time.time() > deadline - 10:
            break
        adj, count = degree_greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 3: Intensive local search on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        # Multiple rounds of improvement
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap2(adj, count, max_iters=100)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 4: Restart from best with perturbations
    if best_adj is not None and time.time() < deadline - 15:
        for i in range(15):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            count = best_count
            
            # Perturb: remove 2-4 random edges
            remove_count = random.randint(2, 4)
            for _ in range(remove_count):
                if count <= 0:
                    break
                u = random.randint(0, n - 1)
                mask = adj[u]
                if not mask:
                    continue
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if v < u:
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
            
            # Improve
            adj, count = local_add(adj, count)
            adj, count = local_swap(adj, count, max_iters=200)
            adj, count = local_add(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 5: Final intensive polish
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        count = best_count
        
        # Aggressive final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swaps
        adj, count = local_swap(adj, count, max_iters=500)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    return adj_to_edges(best_adj)
