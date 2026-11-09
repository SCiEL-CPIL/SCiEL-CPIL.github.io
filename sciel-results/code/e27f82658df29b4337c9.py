"""Program for R_5(C4) maximization via deterministic construction and local search."""
import random
import itertools
import numpy as np
from collections import defaultdict
import time

def construct():
    # We will use an optimized local search with incremental cost updates
    # to find a valid coloring for n=26 (Lazebnik-Woldar bound).
    
    def get_edge_idx(i, j, n):
        # Map (i, j) with i < j to index in flat list
        # Index = sum_{k=0}^{i-1} (n - 1 - k) + (j - i - 1)
        # = i*n - i*(i+1)//2 - 1 + j - i - 1 ?
        # Let's just precompute for speed or use a formula.
        # Formula: idx = i * (2*n - i - 1) // 2 + (j - i - 1)
        return i * (2*n - i - 1) // 2 + (j - i - 1)

    def get_edge_from_idx(idx, n):
        # Inverse of get_edge_idx
        # This is slower, so we might just iterate or precompute a list of edges.
        pass

    def optimize_local_search(n, max_iters=100000):
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_edges = len(edges)
        
        # Initialize random coloring
        random.seed(42)
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build neighbor lists for each color
        # neighbors[c][v] = set of neighbors
        neighbors = [defaultdict(set) for _ in range(5)]
        
        # Build reverse map from edge index to (u, v)
        edge_map = {}
        for idx, (u, v) in enumerate(edges):
            edge_map[idx] = (u, v)
            c = colors[idx]
            neighbors[c][u].add(v)
            neighbors[c][v].add(u)
            
        # Calculate initial cost
        # Cost = sum over colors c, sum over pairs (u, v) of max(0, |N_c(u) & N_c(v)| - 1)
        def calc_cost():
            total = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        common = len(neighbors[c][u] & neighbors[c][v])
                        if common > 1:
                            total += common - 1
            return total

        current_cost = calc_cost()
        
        # Precompute list of vertices for faster iteration?
        # We need to update cost when an edge (u, v) changes color from c_old to c_new.
        
        def delta_cost(u, v, c_old, c_new):
            # Remove (u, v) from c_old, add to c_new
            # Update neighbors structures temporarily? No, calculate delta analytically.
            
            delta = 0
            
            # 1. Effect on c_old
            # u loses v, v loses u
            # Pairs affected: (u, x) and (v, x) for all x != u, v.
            # Also pair (u, v) itself.
            
            # For pair (u, v):
            # Old common neighbors in c_old: N_c_old(u) & N_c_old(v)
            # Note: v is in N(u), u is in N(v).
            # Does v count as a neighbor of u? Yes.
            # So v is in N(u). u is in N(v).
            # Common neighbors of u and v: {w | w in N(u) and w in N(v)}.
            # v is in N(u) but is v in N(v)? No (no self loops).
            # u is in N(v) but is u in N(u)? No.
            # So u and v are NOT common neighbors of each other.
            # So removing edge (u,v) does NOT change N(u) & N(v) with respect to u and v themselves.
            # Wait. N(u) contains v. N(v) contains u.
            # Intersection N(u) & N(v) contains w such that w~u and w~v.
            # v is not in N(v). So v is not in intersection.
            # u is not in N(u). So u is not in intersection.
            # So the set of common neighbors of u and v does NOT include u or v.
            # Therefore, removing edge (u,v) does NOT change the size of N(u) & N(v).
            # Wait, this is wrong.
            # N(u) is the set of vertices connected to u.
            # If we remove edge (u,v), v is removed from N(u).
            # So N'(u) = N(u) \ {v}.
            # Intersection N'(u) & N(v) = (N(u) \ {v}) & N(v) = (N(u) & N(v)) \ {v}.
            # Is v in N(u) & N(v)?
            # v in N(u) is true.
            # v in N(v) is false.
            # So v is NOT in N(u) & N(v).
            # So N'(u) & N(v) = N(u) & N(v).
            # So the number of common neighbors of u and v does NOT change when we remove edge (u,v)?
            # Let's check.
            # Vertices u, v. Neighbors of u: {v, ...}. Neighbors of v: {u, ...}.
            # Common neighbors: w such that w~u and w~v.
            # If w=v, v~u (yes) but v~v (no). So v is not a common neighbor.
            # If w=u, u~u (no).
            # So yes, removing the edge between u and v does not change the set of common neighbors of u and v.
            # It DOES change the common neighbors of u and any other x.
            # N'(u) = N(u) \ {v}.
            # N'(u) & N(x) = (N(u) & N(x)) \ {v}.
            # So if v was in N(x), the count decreases by 1.
            
            # So for c_old:
            # For each x != u, v:
            #   if v in neighbors[c_old][x]:
            #      k_u_x decreases by 1.
            #   if u in neighbors[c_old][x]:
            #      k_v_x decreases by 1.
            
            # We need to update the cost contribution of pairs (u, x) and (v, x).
            
            def cost_term(k):
                return max(0, k - 1)
                
            # Iterate x
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Pair (u, x) in c_old
                # Old k = |N(u) & N(x)|
                # New k = |N(u) & N(x) \ {v}|
                # Decrease by 1 if v in N(x)
                old_k_ux = len(neighbors[c_old][u] & neighbors[c_old][x])
                if v in neighbors[c_old][x]:
                    new_k_ux = old_k_ux - 1
                else:
                    new_k_ux = old_k_ux
                delta += cost_term(new_k_ux) - cost_term(old_k_ux)
                
                # Pair (v, x) in c_old
                old_k_vx = len(neighbors[c_old][v] & neighbors[c_old][x])
                if u in neighbors[c_old][x]:
                    new_k_vx = old_k_vx - 1
                else:
                    new_k_vx = old_k_vx
                delta += cost_term(new_k_vx) - cost_term(old_k_vx)
                
                # Pair (u, v) in c_old
                # As established, common neighbors of u and v don't change.
                # So delta for (u,v) is 0.
                
            # 2. Effect on c_new
            # u gains v, v gains u
            # For each x != u, v:
            #   if v in neighbors[c_new][x]:
            #      k_u_x increases by 1.
            #   if u in neighbors[c_new][x]:
            #      k_v_x increases by 1.
            
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Pair (u, x) in c_new
                old_k_ux = len(neighbors[c_new][u] & neighbors[c_new][x])
                if v in neighbors[c_new][x]:
                    new_k_ux = old_k_ux + 1
                else:
                    new_k_ux = old_k_ux
                delta += cost_term(new_k_ux) - cost_term(old_k_ux)
                
                # Pair (v, x) in c_new
                old_k_vx = len(neighbors[c_new][v] & neighbors[c_new][x])
                if u in neighbors[c_new][x]:
                    new_k_vx = old_k_vx + 1
                else:
                    new_k_vx = old_k_vx
                delta += cost_term(new_k_vx) - cost_term(old_k_vx)
                
            return delta

        # Main loop
        for _ in range(max_iters):
            if current_cost == 0:
                return n, colors, True
            
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            c_old = colors[idx]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            d = delta_cost(u, v, c_old, c_new)
            
            if d < 0:
                # Accept
                current_cost += d
                colors[idx] = c_new
                neighbors[c_old][u].remove(v)
                neighbors[c_old][v].remove(u)
                neighbors[c_new][u].add(v)
                neighbors[c_new][v].add(u)
                
        return n, colors, current_cost == 0

    # Try n=26
    n_out, colors_out, is_valid = optimize_local_search(26, max_iters=50000)
    if is_valid:
        return {'n': 26, 'colors': colors_out}
    
    # Fallback to smaller n if 26 fails
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        n_out, colors_out, is_valid = optimize_local_search(n, max_iters=20000)
        if is_valid:
            return {'n': n, 'colors': colors_out}
            
    return {'n': 11, 'colors': [0]*55}
