import random
import time
import math

def construct():
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def popcount(x):
        # Use bit_count if available (Python 3.10+), else fallback
        if hasattr(int, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def is_valid_addition(u, v):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Constraint: For all distinct a, b, c, |N(a) & N(b) & N(c)| <= 2.
        Adding (u, v) only affects triples containing u or v.
        """
        mask_u = adj[u]
        mask_v = adj[v]
        
        # We need to check if adding v to N(u) and u to N(v) creates a violation.
        # A violation occurs if there exist w, z such that:
        # 1. v is in N(w) and N(z), and |N(u) & N(w) & N(z)| >= 2.
        #    (Because after adding, N(u) will contain v, so intersection becomes |N(u) & N(w) & N(z)| + 1)
        # 2. u is in N(w) and N(z), and |N(v) & N(w) & N(z)| >= 2.
        
        # To speed this up, we can iterate w, z and check conditions.
        # n=20, so n^2 = 400 checks.
        
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[w]
            
            # Check condition 1: v in N(w)
            if (mask_w >> v) & 1:
                # v is neighbor of w.
                # Now check all z > w (z != u, v)
                for z in range(w + 1, n):
                    if z == u or z == v:
                        continue
                    mask_z = adj[z]
                    
                    # Check if v is in N(z)
                    if (mask_z >> v) & 1:
                        # v is common neighbor of w and z.
                        # Check if u has >= 2 common neighbors with w and z.
                        common_u = mask_u & mask_w & mask_z
                        if popcount(common_u) >= 2:
                            return False
                    
                    # Check condition 2: u in N(w) and u in N(z)
                    # We already know u in N(w) is (mask_w >> u) & 1
                    if (mask_w >> u) & 1:
                        if (mask_z >> u) & 1:
                            common_v = mask_v & mask_w & mask_z
                            if popcount(common_v) >= 2:
                                return False
            
            # Optimization: If v is NOT in N(w), condition 1 fails for this w.
            # But condition 2 might still hold if u is in N(w).
            # The logic above handles both independently.
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges

    def get_missing_edges():
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        return missing

    def greedy_fill():
        """
        Try to add all possible missing edges.
        Returns the new edge count.
        """
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_addition(u, v):
                            add_edge(u, v)
                            improved = True
        return edge_count()

    def random_greedy_init():
        nonlocal adj
        adj = [0] * n
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        for u, v in edges_copy:
            if is_valid_addition(u, v):
                add_edge(u, v)
        return greedy_fill()

    def systematic_local_search():
        """
        Perform systematic 1-swap moves (remove one, add one) followed by greedy fill.
        """
        nonlocal adj
        current_count = edge_count()
        improved = True
        
        while improved:
            improved = False
            existing = get_edges()
            missing = get_missing_edges()
            
            # If no missing edges, we are done (or stuck)
            if not missing:
                break
                
            for e_out in existing:
                # If we improved, existing/missing lists are stale. 
                # For simplicity and correctness, if we improve, we break and restart the loop.
                # This is slower but safer.
                for e_in in missing:
                    # Skip if same edge
                    if e_out == e_in:
                        continue
                    
                    u_out, v_out = e_out
                    u_in, v_in = e_in
                    
                    # Remove e_out
                    remove_edge(u_out, v_out)
                    
                    # Try to add e_in
                    if is_valid_addition(u_in, v_in):
                        add_edge(u_in, v_in)
                        
                        # Try to fill more
                        new_count = greedy_fill()
                        
                        if new_count > current_count:
                            current_count = new_count
                            improved = True
                            # Break out to refresh lists
                            break
                        else:
                            # Revert e_in
                            remove_edge(u_in, v_in)
                    
                    # Re-add e_out
                    add_edge(u_out, v_out)
                
                if improved:
                    break
            
        return current_count

    def perturb_and_refill(k=2):
        """
        Remove k random edges and refill.
        """
        nonlocal adj
        existing = get_edges()
        if len(existing) < k:
            return edge_count()
        
        to_remove = random.sample(existing, k)
        for u, v in to_remove:
            remove_edge(u, v)
            
        return greedy_fill()

    # Main Execution
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_count = 0
    best_adj = [0] * n
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Initialize
        count = random_greedy_init()
        
        # 2. Local Search
        count = systematic_local_search()
        
        # 3. Update Best
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # 4. If we have time, try to improve the best found via perturbation
        # But systematic_local_search is expensive. 
        # Let's just do a few more restarts if time permits.
        # Or, perturb the current best and refill.
        
        if time.time() < deadline - 10:
            # Perturb the best found
            adj = best_adj[:]
            count = perturb_and_refill(3)
            count = systematic_local_search()
            if count > best_count:
                best_count = count
                best_adj = adj[:]

    # Final Polish on the best
    adj = best_adj[:]
    final_count = greedy_fill()
    if final_count > best_count:
        best_count = final_count
        best_adj = adj[:]
        
    # Reconstruct edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    return best_edges
