import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_edges = len(all_edges)

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if adj[u] & (1 << v):
            return False
        # For K3,3: no 3 vertices can have 3 common neighbors
        # Adding (u,v) creates new common neighbors between {u,v} and any pair
        # Check: for any a,b not in {u,v}, if u and v are both neighbors of a and b,
        # then {a,b} would have u,v as common neighbors. Need to check if there's a 3rd.
        # More precisely: after adding (u,v), for any triple {x,y,z}, they must not have 3 common neighbors.
        # The new common neighbors involving u and v:
        # - u is now neighbor of v, so for any a,b where v is neighbor of both a and b,
        #   u becomes a common neighbor of {a,b,v}... 
        # 
        # Simpler check: After adding (u,v), for every pair (a,b) where a,b != u,v:
        #   common(a,b) = adj[a] & adj[b] (after update)
        #   We need |common(a,b)| <= 2 for all pairs
        # But that's expensive. Instead, check the critical cases:
        # 
        # When we add (u,v), the only new "common neighbor" relationships involve u or v.
        # Specifically, for any a where v is a neighbor of a:
        #   u becomes a common neighbor of {v, a, x} for any x where v is also neighbor of x
        #   So we need: for any a,b where v is neighbor of both, |common(u,a,b) ∪ {v}| <= 2
        #   i.e., |adj[u] & adj[a] & adj[b] & ~(1<<v) & ~(1<<u)| + 1 <= 2
        #   i.e., |adj[u] & adj[a] & adj[b] & ~(1<<v) & ~(1<<u)| <= 1
        #
        # Similarly for u's neighbors.
        
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Check triples involving v and two neighbors of v
        # For any a,b in N(v)\{u}, need |N(u) ∩ N(a) ∩ N(b) \ {v}| <= 1
        # (since v will be a common neighbor of {u,a,b} after adding edge)
        
        # Get neighbors of v (excluding u)
        nv_list = []
        temp = mask_v
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            temp ^= lsb
        
        if len(nv_list) >= 2:
            for i in range(len(nv_list)):
                for j in range(i + 1, len(nv_list)):
                    a = nv_list[i]
                    b = nv_list[j]
                    # Common neighbors of u, a, b excluding v
                    common = mask_u & adj[a] & adj[b] & ~(1 << v)
                    if common & (common - 1):  # more than 1 bit set
                        return False
        
        # Check triples involving u and two neighbors of u
        nu_list = []
        temp = mask_u
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            nu_list.append(idx)
            temp ^= lsb
        
        if len(nu_list) >= 2:
            for i in range(len(nu_list)):
                for j in range(i + 1, len(nu_list)):
                    a = nu_list[i]
                    b = nu_list[j]
                    common = mask_v & adj[a] & adj[b] & ~(1 << u)
                    if common & (common - 1):
                        return False
        
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_fill(adj, deadline):
        """Greedily add missing edges in random order."""
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, deadline):
        """Local search with kicks: remove random edges, greedily refill."""
        current = adj[:]
        best = adj[:]
        best_count = count_edges(current)
        current_count = best_count
        
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < deadline - 1:
            improved = False
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current[i] & (1 << j))]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline - 1:
                        break
                    if can_add(current, u, v):
                        current[u] |= (1 << v)
                        current[v] |= (1 << u)
                        current_count += 1
                        improved = True
            
            if current_count > best_count:
                best_count = current_count
                best = current[:]
                no_improve = 0
            elif improved:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                # Kick: remove k random edges, then greedily refill
                k = random.randint(2, 6)
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if current[i] & (1 << j)]
                if not existing:
                    break
                to_remove = random.sample(existing, min(k, len(existing)))
                for u, v in to_remove:
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
                    current_count -= 1
                
                # Greedily refill
                current = greedy_fill(current, deadline)
                current_count = count_edges(current)
                
                if current_count > best_count:
                    best_count = current_count
                    best = current[:]
                    no_improve = 0
                else:
                    no_improve = max_no_improve  # Reset to trigger another kick soon
            
            if not improved and no_improve >= max_no_improve * 2:
                break
        
        return best, best_count

    best_adj = [0] * n
    best_count = 0
    
    # Multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        random.seed(num_restarts * 7919)  # Different seed each restart
        
        adj = [0] * n
        
        # Greedy initial construction
        pairs = all_edges[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Greedy fill
        adj = greedy_fill(adj, deadline - 2)
        
        # Local search
        adj, count = local_search(adj, deadline - 2)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
