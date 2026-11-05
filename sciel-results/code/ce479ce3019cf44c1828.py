import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def score(adj):
        """Return max degree sum over all triangles."""
        degs = [bin(adj[i]).count('1') for i in range(n)]
        best = 0
        # Find all triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    # common neighbors of u and v
                    common = adj[u] & adj[v]
                    # remove u and v themselves
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # iterate over common neighbors w > v to avoid duplicates
                    # We only care about max sum, so we can just check all w in common
                    # But we must avoid double counting triangles? 
                    # Actually, for max sum, we just need to check every triangle once.
                    # The loop u < v < w ensures each triangle is visited once.
                    w = v + 1
                    while w < n:
                        if common & (1 << w):
                            s = degs[u] + degs[v] + degs[w]
                            if s > best:
                                best = s
                        w += 1
        return best
    
    def calculate_score_fast(adj):
        """
        Optimized score calculation using bit operations to find triangles.
        We iterate over edges (u,v) and check common neighbors.
        To avoid duplicates, we only consider triangles where u < v < w.
        """
        degs = [bin(adj[i]).count('1') for i in range(n)]
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # We only care about w > v to ensure u < v < w
                    # Mask out bits <= v
                    mask = (1 << (v + 1)) - 1
                    common &= ~mask
                    while common:
                        # Extract lowest set bit
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    # Initial: K_{8,8} + edge (0,1)
    # This is a bipartite graph plus one edge, creating one triangle (0,1,8) if 8 is in other part?
    # Wait, K_{8,8} vertices 0..7 and 8..15. Edge (0,1) is within part 0.
    # Triangles involve (0,1) and any common neighbor. 0 is connected to 8..15. 1 is connected to 8..15.
    # So (0,1,w) is a triangle for w in 8..15.
    # deg(0) = 8 (to 8..15) + 1 (to 1) = 9? No, in K_{8,8}, deg is 8.
    # Add edge (0,1): deg(0)=9, deg(1)=9. deg(w)=8.
    # Sum = 9+9+8 = 26.
    # This is the baseline score 26? Or 24?
    # The prompt says "Scores below 24 would beat...".
    # Let's check the initial score.
    
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj = make_adj(edges)
    current_score = score(adj)
    
    best_score = current_score
    best_adj = adj[:]
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Simulated annealing with edge swaps
    random.seed(42)
    
    def get_random_edge(adj):
        """Pick a random edge (u, v) with u < v."""
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and (adj[u] & (1 << v)):
                return u, v

    def get_random_non_edge(adj):
        """Pick a random non-edge (u, v) with u < v."""
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and not (adj[u] & (1 << v)):
                return u, v

    for restart in range(50):
        if not time_ok():
            break
        
        # For restarts > 0, start from best_adj with some perturbation
        if restart == 0:
            adj = make_adj(edges)
        else:
            adj = best_adj[:]
            # Perturb: do a few random swaps
            for _ in range(random.randint(5, 20)):
                ru, rv = get_random_edge(adj)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                au, av = get_random_non_edge(adj)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
        
        current_score = calculate_score_fast(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
        
        # SA loop
        T = 5.0
        T_min = 0.1
        steps_per_temp = 500
        
        while T > T_min and time_ok():
            for step in range(steps_per_temp):
                if not time_ok():
                    break
                
                # Pick a random edge to remove
                ru, rv = get_random_edge(adj)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                # Pick a random non-edge to add
                au, av = get_random_non_edge(adj)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                
                new_score = calculate_score_fast(adj)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                else:
                    # Undo
                    adj[au] &= ~(1 << av)
                    adj[av] &= ~(1 << au)
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
            
            T *= 0.95
        
        # Robust Polish Phase: Deterministic Local Search
        # Iteratively try all single swaps. 
        # To handle state correctly, we snapshot the edge list, 
        # try a swap, check score, and revert if not improving.
        # We repeat until no improvement is found or time runs out.
        improved = True
        while improved and time_ok():
            improved = False
            # List current edges and non-edges
            cur_edges = []
            non_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        cur_edges.append((u, v))
                    else:
                        non_edges.append((u, v))
            
            base_score = calculate_score_fast(adj)
            
            for e1 in cur_edges:
                if not time_ok(): 
                    break
                u1, v1 = e1
                # Remove e1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                for e2 in non_edges:
                    if not time_ok(): 
                        break
                    u2, v2 = e2
                    # Add e2
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    s = calculate_score_fast(adj)
                    
                    # Revert add e2
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    
                    if s < base_score:
                        # Accept swap: e1 is already removed, so just keep e2 added?
                        # No, we reverted e2. So we must re-add e2 and leave e1 removed.
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        base_score = s
                        improved = True
                        if s < best_score:
                            best_score = s
                            best_adj = adj[:]
                        
                        # Break out of both loops to refresh edge lists
                        break
                
                if improved:
                    break
                    # Note: e1 is currently removed from adj.
                    # If we broke out of inner loop due to improvement, e1 is still removed.
                    # This is correct for the next iteration's state.
                    # However, if we broke out of inner loop due to time, e1 is still removed!
                    # We must ensure e1 is restored if we didn't accept the swap.
            
            # If we didn't improve in the inner loop, e1 is still removed.
            # We need to restore e1 if we didn't accept a swap involving it.
            # But wait, the logic above:
            # 1. Remove e1
            # 2. Loop e2:
            #    a. Add e2
            #    b. Eval
            #    c. Revert e2
            #    d. If improve: Re-add e2, set improved=True, break inner loop.
            # 3. If improved, break outer loop.
            
            # What if we finished the inner loop without improving?
            # Then e1 is still removed. We must add it back.
            if not improved:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)

        if best_score <= 23:
            break
    
    # Convert best_adj to edge list
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
    
    # Verify and Repair if necessary
    if len(final_edges) != target_edges:
        # This should not happen with the fixed logic, but just in case,
        # we reconstruct from the initial known-good state if it fails.
        # However, it's better to just return the best valid graph we have.
        # If best_adj is corrupted, we should have kept a separate valid copy.
        # Let's do a final sanity check on best_adj.
        edge_count = sum(bin(best_adj[i]).count('1') for i in range(n)) // 2
        if edge_count != target_edges:
            # Fallback to initial construction
            edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
            final_edges = edges
        else:
            # Re-verify edge list matches best_adj
            pass

    return {'n': 16, 'edges': final_edges}
