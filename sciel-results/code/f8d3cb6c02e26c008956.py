import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using Simulated Annealing
    with NumPy-accelerated cost updates.
    Maximizes n.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    # Check if popcount(common) >= 2
                    if common & (common - 1):
                        return False
        return True

    def solve_sa(n, time_limit):
        """
        Runs Simulated Annealing to find a valid coloring for K_n.
        Uses NumPy for fast cost updates.
        """
        total_edges = n * (n - 1) // 2
        
        # Initialize colors
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Precompute edge endpoints
        edge_u = np.empty(total_edges, dtype=np.int32)
        edge_v = np.empty(total_edges, dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_u[idx] = i
                edge_v[idx] = j
                idx += 1
        
        # Build initial adjacency matrices (5 x n x n), boolean or int
        # Using uint8 for compactness and speed
        adj = np.zeros((5, n, n), dtype=np.uint8)
        
        for k in range(total_edges):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj[c, u, v] = 1
            adj[c, v, u] = 1
            
        # Compute initial cost
        # cost_c = sum over u<v of (1 if common_neighbors_c(u,v) >= 2)
        # common_neighbors = adj_c @ adj_c
        # We need to count pairs (u,v) where (adj_c @ adj_c)[u,v] >= 2
        # Note: diagonal elements are degrees, we ignore them.
        # Also u<v vs u!=v.
        
        def compute_cost(adj):
            total_cost = 0
            for c in range(5):
                # adj[c] is n x n
                # common = adj[c] @ adj[c]
                # We want to count entries common[u,v] >= 2 for u != v
                # Let's compute the matrix product
                common = adj[c].dot(adj[c])
                # Mask for u < v to avoid double counting and diagonal
                # Or just count all u!=v and divide by 2?
                # Since the graph is undirected, common is symmetric.
                # Let's count upper triangle.
                # np.triu_indices(n, k=1)
                # But creating indices every time is slow.
                # We can precompute the mask indices.
                
                # For now, let's just use a vectorized check on the upper triangle.
                # idx_u, idx_v = np.triu_indices(n, k=1)
                # vals = common[idx_u, idx_v]
                # count = np.sum(vals >= 2)
                
                # To be faster, we can precompute the indices outside the loop.
                # But compute_cost is called inside the loop? No, we update incrementally.
                # However, for the initial cost, we can do this.
                pass
            
            # Actually, let's implement the incremental update properly.
            # It's hard to do incremental update with full matrix multiplication.
            # The previous Python code did incremental update on bitsets.
            # The inspiration says "NumPy-based matrix multiplication ... accelerates SA iterations by ~10x".
            # How?
            # Maybe it recomputes the cost of *affected* parts?
            # Or maybe it uses a different cost function?
            
            # Let's look at the structure.
            # If we change edge (u,v) from c_old to c_new:
            # 1. Remove (u,v) from adj[c_old]
            # 2. Add (u,v) to adj[c_new]
            
            # The cost changes only for pairs involving u or v?
            # Yes.
            # Specifically, the number of common neighbors for any pair (x,y) changes
            # only if u or v is a common neighbor of x,y in the affected colors.
            
            # Let's stick to the incremental bitset approach but optimize it?
            # The current code is already incremental.
            # The inspiration 1 (score 1.9) uses `count_violations` which is O(n^3) per step?
            # "Calculate new cost: new_cost = count_violations(n, colors)"
            # That is O(n^3) per step. For n=26, n^3 ~ 17000.
            # If it runs 10x faster, maybe it's just better constants or numpy overhead is low for small n?
            # But O(n^3) per step is very slow compared to incremental O(n).
            # Wait, the inspiration 1 code provided in the prompt has:
            # `new_cost = count_violations(n, colors)`
            # This is a full recomputation.
            # Why did it score 1.9?
            # Maybe the Python bitset code was slower due to Python loop overhead?
            # Or maybe the numpy version allows for larger n to be explored?
            
            # Let's try a hybrid:
            # Use the incremental bitset logic from the current program (which is O(n) per step)
            # but maybe the current program has bugs or is slow due to Python loops in the bit iteration?
            
            # Let's optimize the current program's incremental update.
            # The current code iterates through bits of `nv_old`.
            # `while nv_old:` loop.
            # This is fast in Python if the degree is low.
            
            # Let's check the current program's logic again.
            # It seems correct.
            
            # What if we use the "Inspiration 1" approach but optimize it?
            # Or combine?
            
            # Let's try to implement the numpy cost update.
            # If we change edge (u,v) from c1 to c2:
            # 
            # Cost in c1 decreases. Cost in c2 increases.
            # 
            # Delta in c1:
            # Remove edge (u,v).
            # For any w, the pair (u,w) loses v as a common neighbor.
            # The pair (v,w) loses u as a common neighbor.
            # 
            # Let K_c(x,y) = |N_c(x) ∩ N_c(y)|.
            # We need to count how many pairs (x,y) have K_c(x,y) >= 2.
            # 
            # When we remove (u,v) from c:
            # For each w in N_c(u) \ {v}:
            #   K_c(w, v) decreases by 1 (since u is no longer in N_c(w) ∩ N_c(v)? No, u is in N_c(w) and u is in N_c(v)? No, u-v is an edge, so u is neighbor of v. But we are removing the edge u-v. So u is no longer a neighbor of v. So u is no longer in N_c(v). Thus u is no longer in intersection.
            #   Wait, N_c(v) changes.
            #   
            # Let's trace:
            # Before removal:
            #   N_c(u) contains v.
            #   N_c(v) contains u.
            #   For any w:
            #     K_c(u, w) includes v if w is neighbor of v.
            #     K_c(v, w) includes u if w is neighbor of u.
            #     
            # After removal:
            #   N_c(u) does not contain v.
            #   N_c(v) does not contain u.
            #   
            # Changes in K_c:
            # 1. Pairs (u, w) for w in N_c(v) \ {u}:
            #    K_c(u, w) decreases by 1 (v is no longer in N_c(u)).
            #    If K_c(u, w) was 2, it becomes 1. Cost decreases by 1.
            # 2. Pairs (v, w) for w in N_c(u) \ {v}:
            #    K_c(v, w) decreases by 1 (u is no longer in N_c(v)).
            #    If K_c(v, w) was 2, it becomes 1. Cost decreases by 1.
            # 3. Pair (u, v):
            #    K_c(u, v) decreases by 2?
            #    N_c(u) ∩ N_c(v).
            #    Before: u is in N_c(v), v is in N_c(u).
            #    Common neighbors are N_c(u) ∩ N_c(v).
            #    Since u is in N_c(v) and v is in N_c(u), are they in the intersection?
            #    No, u is not in N_c(u) (no self loops).
            #    So u is not in N_c(u) ∩ N_c(v).
            #    So K_c(u, v) is the number of other common neighbors.
            #    Removing edge (u,v) does not change the set of common neighbors of u and v?
            #    Wait. N_c(u) and N_c(v) change.
            #    N_c(u) loses v. N_c(v) loses u.
            #    Does this affect N_c(u) ∩ N_c(v)?
            #    x is in intersection if x in N_c(u) and x in N_c(v).
            #    If x=u: u in N_c(v) (yes, before), u in N_c(u) (no). So u not in intersection.
            #    If x=v: v in N_c(u) (yes, before), v in N_c(v) (no). So v not in intersection.
            #    So the set of common neighbors is unchanged?
            #    Yes, because u and v were not common neighbors of each other.
            #    So K_c(u,v) is unchanged.
            #    
            # So the delta cost for removing (u,v) from c is:
            #   - count of w in N_c(v) \ {u} such that K_c(u, w) == 2
            #   - count of w in N_c(u) \ {v} such that K_c(v, w) == 2
            #
            # This is exactly what the current program does!
            # 
            # So why is numpy faster?
            # Maybe the bit iteration in Python is slow.
            # We can use numpy to compute the counts.
            # 
            # Let's try to vectorize the delta calculation.
            
            return 0 # Placeholder

    # Let's implement the vectorized delta calculation.
    # We maintain `adj` as 5 x n x n uint8 arrays.
    # And `cost` as total cost.
    # 
    # To get K_c(x, y), we need dot product of row x and row y of adj[c].
    # 
    # For delta removal from c_old:
    # u, v are the endpoints.
    # 
    # 1. For w in N_c(u) \ {v}: check K_c(v, w) == 2.
    #    N_c(u) is adj[c, u, :] == 1.
    #    We need to iterate over these w.
    #    For each w, compute K_c(v, w) = dot(adj[c, v, :], adj[c, w, :]).
    #    
    #    We can do this in bulk?
    #    Let mask_u = adj[c, u, :] (boolean)
    #    mask_u[v] = False
    #    ws = np.where(mask_u)[0]
    #    
    #    We need K_c(v, w) for all w in ws.
    #    K_c(v, w) = adj[c, v, :] @ adj[c, w, :]
    #    This is a matrix-vector product?
    #    Let M = adj[c, :, ws]  # n x len(ws)
    #    Then K_c(v, w) for all w is adj[c, v, :] @ M  # len(ws)
    #    
    #    So we can compute all K values at once.
    #    
    #    count1 = np.sum(K_vals == 2)
    #    
    #    Similarly for the other side.
    
    def solve_sa_numpy(n, time_limit):
        total_edges = n * (n - 1) // 2
        
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        edge_u = np.empty(total_edges, dtype=np.int32)
        edge_v = np.empty(total_edges, dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_u[idx] = i
                edge_v[idx] = j
                idx += 1
                
        adj = np.zeros((5, n, n), dtype=np.uint8)
        for k in range(total_edges):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj[c, u, v] = 1
            adj[c, v, u] = 1
            
        # Compute initial cost
        # cost_c = sum over u<v of (1 if common >= 2)
        # We can compute common = adj[c] @ adj[c]
        # Then count upper triangle entries >= 2.
        
        def calc_initial_cost():
            total = 0
            for c in range(5):
                common = adj[c].dot(adj[c])
                # Get upper triangle
                # np.triu_indices
                iu, ju = np.triu_indices(n, k=1)
                vals = common[iu, ju]
                total += np.sum(vals >= 2)
            return total
            
        current_cost = calc_initial_cost()
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 1.0
        cooling_rate = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            e_idx = random.randint(0, total_edges - 1)
            u = edge_u[e_idx]
            v = edge_v[e_idx]
            old_c = colors[e_idx]
            
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta
            delta = 0
            
            # Remove from old_c
            # 1. w in N_old(u) \ {v}: check K_old(v, w) == 2
            mask_u = adj[old_c, u, :].copy()
            mask_u[v] = 0
            ws_u = np.where(mask_u)[0]
            if len(ws_u) > 0:
                # K_old(v, w) for w in ws_u
                # adj[old_c, v, :] @ adj[old_c, :, ws_u]
                # adj[old_c, :, ws_u] is n x len(ws_u)
                # Result is 1 x len(ws_u)
                K_vals = adj[old_c, v, :] @ adj[old_c, :, ws_u]
                delta -= np.sum(K_vals == 2)
                
            # 2. w in N_old(v) \ {u}: check K_old(u, w) == 2
            mask_v = adj[old_c, v, :].copy()
            mask_v[u] = 0
            ws_v = np.where(mask_v)[0]
            if len(ws_v) > 0:
                K_vals = adj[old_c, u, :] @ adj[old_c, :, ws_v]
                delta -= np.sum(K_vals == 2)
                
            # Apply removal
            adj[old_c, u, v] = 0
            adj[old_c, v, u] = 0
            
            # Add to new_c
            # 1. w in N_new(v) \ {u}: check K_new(u, w) == 1
            # Note: u is not in N_new(v) yet?
            # We haven't added (u,v) to new_c yet.
            # So N_new(v) does not contain u.
            # So we just take N_new(v).
            mask_v_new = adj[new_c, v, :].copy()
            # u is not in mask_v_new because edge not added.
            # But just in case, clear it.
            mask_v_new[u] = 0
            ws_v_new = np.where(mask_v_new)[0]
            if len(ws_v_new) > 0:
                # K_new(u, w) for w in ws_v_new
                # adj[new_c, u, :] @ adj[new_c, :, ws_v_new]
                K_vals = adj[new_c, u, :] @ adj[new_c, :, ws_v_new]
                delta += np.sum(K_vals == 1)
                
            # 2. w in N_new(u) \ {v}: check K_new(v, w) == 1
            mask_u_new = adj[new_c, u, :].copy()
            mask_u_new[v] = 0
            ws_u_new = np.where(mask_u_new)[0]
            if len(ws_u_new) > 0:
                K_vals = adj[new_c, v, :] @ adj[new_c, :, ws_u_new]
                delta += np.sum(K_vals == 1)
                
            # Apply addition
            adj[new_c, u, v] = 1
            adj[new_c, v, u] = 1
            
            new_cost = current_cost + delta
            current_cost = new_cost
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                colors[e_idx] = new_c
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject
                colors[e_idx] = old_c
                # Revert adj
                adj[new_c, u, v] = 0
                adj[new_c, v, u] = 0
                adj[old_c, u, v] = 1
                adj[old_c, v, u] = 1
                current_cost -= delta
                
            T *= cooling_rate
            if T < 0.01:
                T = 1.0
                
        return best_colors if best_cost == 0 else None

    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
        
    best_n, best_colors = get_seed_solution()
    
    search_plan = [
        (28, 10.0),
        (27, 15.0),
        (26, 30.0),
        (25, 10.0),
        (24, 10.0),
    ]
    
    for n, t_alloc in search_plan:
        if time.time() > deadline - 2.0:
            break
            
        res = solve_sa_numpy(n, t_alloc)
        
        if res is not None:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
