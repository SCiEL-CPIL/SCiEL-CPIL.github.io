import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 140  # Leave margin for 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding edge (u,v) to adj is valid (no K2,3)
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        # Check: |N(u) ∩ N(v)| <= 2
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # For each w in N(u), |N(w) ∩ N(v)| <= 1
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if (adj[w] & adj[v]).bit_count() > 1:
                return False
            mask ^= lb
        
        # For each w in N(v), |N(w) ∩ N(u)| <= 1
        mask = adj[v]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if (adj[w] & adj[u]).bit_count() > 1:
                return False
            mask ^= lb
        
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
    def greedy_init():
        adj = [0] * n
        count = 0
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        for u, v in edge_list:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        return adj, count

    # Local search: try adding missing edges until no more can be added
    def greedy_add(adj, count):
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

    # Systematic 1-Opt: Try removing each edge and see if it allows adding more
    def systematic_1_opt(adj, count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Iterate through edges in random order to avoid bias
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            for u, v in edge_list:
                if time.time() > deadline - 2:
                    break
                if not (adj[u] & (1 << v)):
                    continue
                
                # Save state
                adj_copy = adj[:]
                
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
                
                # Try to add edges
                new_count = count
                # We can run a quick greedy add
                # To speed up, we can just scan once or twice
                # Full greedy_add might be too slow if called 190 times per pass
                # Let's do a limited greedy add: scan all edges once
                edge_list_add = all_edges[:]
                random.shuffle(edge_list_add)
                for a, b in edge_list_add:
                    if adj[a] & (1 << b):
                        continue
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        new_count += 1
                
                if new_count > count + 1: # Net gain
                    count = new_count
                    improved = True
                else:
                    # Restore
                    adj = adj_copy
                    # count is already restored implicitly if we don't update it, 
                    # but we modified adj in place. 
                    # Wait, adj_copy was made before removal.
                    # So restoring adj = adj_copy restores the graph.
                    # count was decremented, so we need to restore count too.
                    # But in the loop, count is a local variable.
                    # We only update count if improved.
                    # So if not improved, count remains as it was before removal?
                    # No, we did count -= 1.
                    # So we must restore count.
                    count += 1
        
        return adj, count

    # Perturbation: remove random edges
    def perturb(adj, count, num_remove):
        for _ in range(num_remove):
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
        return adj, count

    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Multiple random starts
    start_time = time.time()
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
        adj, count = greedy_init()
        adj, count = greedy_add(adj, count)
        
        # Polish with 1-opt
        adj, count = systematic_1_opt(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Iterative improvement on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        while time.time() < deadline - 5:
            # Polish
            adj, count = greedy_add(adj, count)
            adj, count = systematic_1_opt(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            # Perturb and restart
            # Remove 2-5 edges
            remove_num = random.randint(2, 5)
            adj, count = perturb(adj, count, remove_num)
            
            # Recover
            adj, count = greedy_add(adj, count)
            adj, count = systematic_1_opt(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]

    # Final polish
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        adj, count = greedy_add(adj, count)
        adj, count = systematic_1_opt(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    if best_adj is None:
        return []
    
    return adj_to_edges(best_adj)
