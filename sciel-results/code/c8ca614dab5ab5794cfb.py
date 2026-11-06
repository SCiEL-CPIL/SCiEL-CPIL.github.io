import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # --- Step 1: Deterministic Construction for n=25 ---
    # Vertices are (x, y) where x, y in {0, 1, 2, 3, 4}
    # Map (x, y) to index x * 5 + y
    # Edge color is determined by the slope of the line connecting (x1, y1) and (x2, y2)
    # Slopes in F_5:
    # If x1 == x2, slope is infinity -> color 0
    # Else, slope = (y2 - y1) * (x2 - x1)^(-1) mod 5
    # The possible slopes are 0, 1, 2, 3, 4.
    # We map slope s to color s.
    
    n_base = 25
    q = 5
    
    def get_idx_base(i, j):
        # i, j are indices 0..24
        # In lexicographic order for n=25
        # idx = i * (2*n - i - 1) // 2 + (j - i - 1)
        return i * (2 * n_base - i - 1) // 2 + (j - i - 1)
    
    def mod_inv(a, mod):
        # a is in 1..4, mod is 5
        for x in range(1, mod):
            if (a * x) % mod == 1:
                return x
        return 0 # Should not happen for prime mod and non-zero a
    
    colors_base = [0] * (n_base * (n_base - 1) // 2)
    
    for i in range(n_base):
        x1 = i // q
        y1 = i % q
        for j in range(i + 1, n_base):
            x2 = j // q
            y2 = j % q
            
            if x1 == x2:
                color = 0  # Infinite slope
            else:
                dx = (x2 - x1) % q
                dy = (y2 - y1) % q
                inv_dx = mod_inv(dx, q)
                slope = (dy * inv_dx) % q
                color = slope
            
            idx = get_idx_base(i, j)
            colors_base[idx] = color
    
    # Verify n=25 is valid (should be, but just in case)
    # We assume it is valid based on theory. 
    # A C4 in color c would imply two vertices with 2 common neighbors in color c.
    # In AG(2,5), lines of a fixed slope form a parallel class.
    # Any two distinct lines in a parallel class are disjoint.
    # A C4 would require two vertices u, v to share two neighbors w, z.
    # This implies edges (u,w), (v,w), (u,z), (v,z) all have color c.
    # This means u, v, w, z form a parallelogram in the affine plane?
    # Actually, if (u,w) and (v,z) have slope c, and (u,z) and (v,w) have slope c...
    # Wait, the color is the slope of the edge.
    # If all 4 edges have slope c, then (u,w) || (v,z) and (u,z) || (v,w).
    # This forms a parallelogram.
    # In a finite affine plane AG(2,q), do parallelograms exist? Yes.
    # Example: (0,0), (1,0), (0,1), (1,1).
    # Edges:
    # (0,0)-(1,0): slope 0.
    # (0,1)-(1,1): slope 0.
    # (0,0)-(0,1): slope inf (color 0).
    # (1,0)-(1,1): slope inf (color 0).
    # So (0,0) and (1,1) share neighbors (1,0) and (0,1) via color 0?
    # Let's check neighbors of (0,0) in color 0:
    # Lines through (0,0) with slope 0: y=0. Points: (0,0), (1,0), (2,0), (3,0), (4,0).
    # Lines through (0,0) with slope inf: x=0. Points: (0,0), (0,1), (0,2), (0,3), (0,4).
    # So N_0(0,0) = {(1,0), (2,0), (3,0), (4,0), (0,1), (0,2), (0,3), (0,4)}.
    # Let's check neighbors of (1,1) in color 0:
    # Lines through (1,1) with slope 0: y=1. Points: (0,1), (1,1), (2,1), (3,1), (4,1).
    # Lines through (1,1) with slope inf: x=1. Points: (1,0), (1,1), (1,2), (1,3), (1,4).
    # N_0(1,1) = {(0,1), (2,1), (3,1), (4,1), (1,0), (1,2), (1,3), (1,4)}.
    # Common neighbors: (0,1) and (1,0).
    # So (0,0) and (1,1) have TWO common neighbors in color 0: (0,1) and (1,0).
    # This IS a C4! (0,0)-(0,1)-(1,1)-(1,0)-(0,0).
    # Wait, the condition is "No colour class may contain a four-cycle".
    # A four-cycle in the graph of color 0 means 4 vertices a,b,c,d such that (a,b), (b,c), (c,d), (d,a) are all color 0.
    # In my check above:
    # Edge (0,0)-(0,1) is color 0.
    # Edge (0,1)-(1,1) is color 0.
    # Edge (1,1)-(1,0) is color 0.
    # Edge (1,0)-(0,0) is color 0.
    # Yes, this is a monochromatic C4.
    
    # CONCLUSION: The simple slope construction on AG(2,5) DOES contain C4s.
    # Why? Because the color classes are not just "lines", but the union of all lines of a specific slope?
    # No, the color of an edge is the slope.
    # The graph G_c consists of edges with slope c.
    # For c=0 (horizontal and vertical? No, slope 0 is horizontal, inf is vertical? 
    # My mapping:
    # x1==x2 -> color 0.
    # Else slope = dy/dx.
    # So color 0 includes vertical lines (inf) AND horizontal lines (slope 0)?
    # No, in my code:
    # if x1 == x2: color = 0.
    # else: color = slope.
    # So color 0 = Vertical lines.
    # Color 1 = Slope 1 lines.
    # ...
    # Color 4 = Slope 4 lines.
    
    # Let's re-evaluate the C4 check for Color 0 (Vertical lines).
    # Edges in Color 0 are vertical segments.
    # A C4 in Color 0 requires 4 vertices forming a cycle using only vertical edges.
    # Vertical edges connect (x, y) and (x, y').
    # To form a cycle, you must change x. But you can only use vertical edges.
    # So you can only move within a column.
    # A cycle in a set of disjoint paths (columns) is impossible.
    # So Color 0 is C4-free.
    
    # What about Color 1 (Slope 1)?
    # Edges connect (x,y) to (x+dx, y+dx).
    # This graph is a union of cycles?
    # In AG(2,5), lines of slope 1 are parallel.
    # The edges of a specific line form a path (or cycle if we wrap around? No, it's a line in the affine plane, not a torus).
    # The graph formed by all edges of slope 1 is a union of 5 disjoint paths (the 5 lines of slope 1).
    # Each line has 5 vertices. The edges connect consecutive vertices along the line.
    # A path of 5 vertices has no cycles.
    # Therefore, no C4.
    
    # So the construction IS valid. My previous counter-example failed because I assumed Color 0 included both horizontal and vertical.
    # In my code, Color 0 is ONLY vertical.
    # Slope 0 (horizontal) gets Color 0?
    # Let's check the code:
    # if x1 == x2: color = 0.
    # else: slope = (dy * inv_dx) % q.
    # If dx != 0 and dy == 0, slope = 0.
    # So Horizontal lines get Color 0.
    # Vertical lines get Color 0.
    # So Color 0 contains BOTH horizontal and vertical edges.
    # THAT creates the C4 (the unit square).
    
    # FIX: We need 5 colors.
    # There are 5 slopes: 0, 1, 2, 3, 4.
    # And 1 "infinity" slope.
    # Total 6 directions.
    # We only have 5 colors.
    # We must merge two directions into one color?
    # If we merge, we might create C4s.
    # Alternatively, use a different construction.
    
    # Lazebnik-Woldar construction for R_5(C4) > 26.
    # It is known that R_k(C4) > k^2 + k + 1 for certain k? No.
    # The bound is related to projective planes.
    # For k=5, a projective plane of order 4 has 21 points.
    # A projective plane of order 5 has 31 points.
    # The Lazebnik-Woldar construction uses a specific algebraic structure.
    
    # Let's stick to a simpler valid construction first.
    # If we cannot guarantee n=25 with 5 colors easily without merging slopes (which risks C4s),
    # we can use the known result that R_5(C4) is at least 12? No, current is 11.
    # Actually, the "slope" idea can be fixed by using the 5 slopes 0,1,2,3,4 and treating Vertical as one of them?
    # No, Vertical is distinct.
    
    # Let's try a different geometric construction:
    # Use the vertices of a pentagon? No.
    
    # Let's go back to Local Search.
    # The current program fails to go beyond 11 because random restarts are inefficient.
    # Inspiration 1 uses SA with fast delta updates.
    # Let's implement SA properly for n=12, 13, ... up to a limit.
    # If we find n=12, that's a score of 12/11 = 1.09.
    # If we find n=13, 1.18.
    
    # To beat the inspiration scores (1.09, 1.36), we need to find n >= 15 or use the n=25 construction correctly.
    
    # Is there a valid 5-coloring of K_25?
    # Yes, if we can decompose K_25 into 5 C4-free graphs.
    # The "Slope" construction fails because we have 6 directions and 5 colors.
    # However, if we use the F_5 field, we can map the 6 directions to 5 colors by identifying two directions that do not form C4s together.
    # Horizontal (0) and Vertical (Inf) form C4s (unit squares).
    # Horizontal (0) and Slope 1?
    # Let's check if union of H and S1 has C4.
    # H: (x,y)-(x+1,y). S1: (x,y)-(x+1,y+1).
    # Cycle: (0,0)-H->(1,0)-S1->(2,1)-H->(3,1)-S1->(4,2)... no return.
    # To close a C4, we need a parallelogram.
    # If we use H and V, we get unit squares.
    # If we use H and S1, do we get C4s?
    # A C4 requires edges e1, e2, e3, e4.
    # If e1, e3 are H, they are parallel horizontal.
    # If e2, e4 are S1, they are parallel diagonal.
    # This forms a parallelogram.
    # So ANY pair of distinct slope classes will form C4s if combined.
    # Therefore, we CANNOT merge any two slope classes.
    # We need 6 colors for AG(2,5). We have 5.
    # So AG(2,5) slope construction does NOT work for 5 colors.
    
    # What about AG(2,4)? n=16.
    # Directions in F_4? F_4 has 3 slopes + inf = 4 directions.
    # We have 5 colors. We can use 4 directions and leave one color unused?
    # Or use a different structure.
    
    # Actually, there is a known construction for R_5(C4).
    # The prompt says "Lazebnik and Woldar gives 26".
    # This implies a valid 5-coloring of K_26 exists.
    # It is likely based on a finite field or group structure.
    
    # Since I cannot derive the exact Lazebnik-Woldar construction from scratch in the code time,
    # I will rely on the **Simulated Annealing** from Inspiration 1, but optimized.
    # I will try to find n=12, 13, 14, 15, 16.
    # n=12 is 66 edges. n=16 is 120 edges.
    # SA with delta updates is very fast.
    
    # Strategy:
    # 1. Start with n=12.
    # 2. Use SA to find a valid coloring.
    # 3. If found, try to extend to n=13 by fixing the 12 vertices and optimizing the new 12 edges.
    #    This "incremental" approach is often more successful than restarting from scratch.
    # 4. Continue until time runs out or we hit a wall.
    
    best_n = 11
    # Fallback n=11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback
    
    def calc_violations_fast(adj_mask, n):
        v = 0
        for c in range(5):
            for u in range(n):
                for vrt in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][vrt]
                    cnt = common.bit_count()
                    if cnt > 1:
                        v += cnt - 1
        return v

    def solve_incremental(n_current, colors_current, adj_mask_current, deadline):
        """
        Attempts to extend the coloring from n_current to n_current + 1.
        Returns (success, new_n, new_colors, new_adj_mask)
        """
        n_new = n_current + 1
        num_edges_new = n_new * (n_new - 1) // 2
        
        # New edges are (n_current, j) for j in 0..n_current-1
        # In lexicographic order, these are the LAST n_current edges.
        # Indices: 
        # Old edges: 0 .. num_edges_old - 1
        # New edges: num_edges_old .. num_edges_new - 1
        
        num_edges_old = n_current * (n_current - 1) // 2
        
        # Initialize new colors
        colors_new = colors_current[:] + [random.randint(0, 4) for _ in range(n_current)]
        
        # Initialize new adj_mask
        adj_mask_new = [[0] * n_new for _ in range(5)]
        
        # Copy old masks
        for c in range(5):
            for u in range(n_current):
                adj_mask_new[c][u] = adj_mask_current[c][u]
        
        # Set new vertex masks (initially 0, will be updated by edges)
        # The new vertex is index n_current.
        
        # Build initial state for new edges
        for k in range(n_current):
            u = n_current
            v = k
            # Find index of edge (v, u) in new array
            # v < u
            idx = get_idx(v, u, n_new)
            c = colors_new[idx]
            adj_mask_new[c][u] |= (1 << v)
            adj_mask_new[c][v] |= (1 << u)
            
        # Calculate violations
        current_violations = calc_violations_fast(adj_mask_new, n_new)
        
        if current_violations == 0:
            return True, n_new, colors_new, adj_mask_new
            
        # SA Loop
        # Only optimize the new edges (indices num_edges_old to num_edges_new - 1)
        # This reduces the search space significantly.
        
        temp = 1.0
        cooling = 0.999
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_violations == 0:
                return True, n_new, colors_new, adj_mask_new
                
            # Pick a random new edge
            new_edge_offset = random.randint(0, n_current - 1)
            idx = num_edges_old + new_edge_offset
            u = n_current
            v = new_edge_offset
            
            old_color = colors_new[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            # Calculate delta
            delta = 0
            
            # Remove from old_color
            c = old_color
            # Neighbors of u in c: adj_mask_new[c][u]
            # Neighbors of v in c: adj_mask_new[c][v]
            
            # Affected pairs involving u: (u, w) where w in N_c(v) \ {u}
            # Actually, changing color of (u,v) affects:
            # 1. The pair (u,v) itself.
            # 2. Pairs (u, w) where w is a common neighbor of u and v in c? No.
            #    Changing (u,v) from c to c' changes the common neighbor count for any pair (x,y) 
            #    such that {u,v} intersects {x,y} and the path x-u-v-y or x-v-u-y exists in c.
            
            # Let's use the general delta logic from Inspiration 1.
            # It checks all neighbors.
            
            # Remove from old_color
            mask_v = adj_mask_new[c][v]
            temp_mask = mask_v
            while temp_mask:
                lsb = temp_mask & -temp_mask
                w = lsb.bit_length() - 1
                temp_mask ^= lsb
                if w == u: continue
                
                # Pair (u, w)
                # Before: (u,w) and (v,w) ? No, we are changing (u,v).
                # The violation count is based on common neighbors.
                # For a pair (a,b), violation is max(0, |N(a) & N(b)| - 1).
                # Changing (u,v) color affects N(u) and N(v).
                # Pairs affected:
                # - (u, v) itself.
                # - (u, w) for all w in N_old(v) \ {u}.
                # - (v, w) for all w in N_old(u) \ {v}.
                
                # Pair (u, w):
                # N_old(u) and N_old(w).
                # N_new(u) = N_old(u) - {v} (if v was in N_old(u), which it is, since (u,v) is in c)
                # Wait, (u,v) is in c. So v in N_c(u) and u in N_c(v).
                # So N_old(u) contains v. N_old(w) contains u?
                # If (u,w) is in c, then u in N_c(w) and w in N_c(u).
                # We are removing v from N_c(u).
                # So for pair (u,w):
                # Old common: N_c(u) & N_c(w).
                # New common: (N_c(u) \ {v}) & N_c(w).
                # If v is in N_c(w), then common count decreases by 1.
                # Else, no change.
                
                if (adj_mask_new[c][w] >> u) & 1:
                    # w is connected to u in c
                    n_old = (adj_mask_new[c][u] & adj_mask_new[c][w]).bit_count()
                    n_new = n_old - 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
                
                # Pair (v, w):
                # N_old(v) contains u.
                # N_new(v) = N_old(v) \ {u}.
                # If u is in N_c(w), common count decreases by 1.
                if (adj_mask_new[c][w] >> v) & 1:
                    # w is connected to v in c
                    n_old = (adj_mask_new[c][v] & adj_mask_new[c][w]).bit_count()
                    n_new = n_old - 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
                
            # Update masks for removal
            adj_mask_new[c][u] &= ~(1 << v)
            adj_mask_new[c][v] &= ~(1 << u)
            
            # Add to new_color
            c = new_color
            mask_v = adj_mask_new[c][v]
            temp_mask = mask_v
            while temp_mask:
                lsb = temp_mask & -temp_mask
                w = lsb.bit_length() - 1
                temp_mask ^= lsb
                if w == u: continue
                
                # Pair (u, w)
                if (adj_mask_new[c][w] >> u) & 1:
                    n_old = (adj_mask_new[c][u] & adj_mask_new[c][w]).bit_count()
                    n_new = n_old + 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
                
                # Pair (v, w)
                if (adj_mask_new[c][w] >> v) & 1:
                    n_old = (adj_mask_new[c][v] & adj_mask_new[c][w]).bit_count()
                    n_new = n_old + 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
            
            # Update masks for addition
            adj_mask_new[c][u] |= (1 << v)
            adj_mask_new[c][v] |= (1 << u)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                colors_new[idx] = new_color
            else:
                # Revert
                # Remove from new_color
                adj_mask_new[new_color][u] &= ~(1 << v)
                adj_mask_new[new_color][v] &= ~(1 << u)
                # Add to old_color
                adj_mask_new[old_color][u] |= (1 << v)
                adj_mask_new[old_color][v] |= (1 << u)
                
            temp *= cooling
            
        if current_violations == 0:
            return True, n_new, colors_new, adj_mask_new
        return False, n_current, colors_current, adj_mask_current

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    # Start with n=11 fallback
    n_curr = 11
    colors_curr = best_colors
    adj_curr = [[0] * n_curr for _ in range(5)]
    for i in range(n_curr):
        for j in range(i + 1, n_curr):
            idx = get_idx(i, j, n_curr)
            c = colors_curr[idx]
            adj_curr[c][i] |= (1 << j)
            adj_curr[c][j] |= (1 << i)
            
    # Try to extend to 12, 13, ...
    for target_n in range(12, 27):
        if time.time() > deadline - 5.0:
            break
            
        # Multiple restarts for this target
        found = False
        for restart in range(10):
            if time.time() > deadline - 2.0:
                break
                
            success, n_res, colors_res, adj_res = solve_incremental(n_curr, colors_curr, adj_curr, deadline)
            
            if success:
                n_curr = n_res
                colors_curr = colors_res
                adj_curr = adj_res
                best_n = n_res
                best_colors = colors_res
                found = True
                break
        
        if not found:
            # If we fail to extend, we might still try to solve from scratch for a smaller n?
            # No, incremental is better. If we can't go from 11 to 12, we are stuck at 11.
            # But maybe a different random seed for the new edges works?
            # We already did 10 restarts.
            break

    return {'n': best_n, 'colors': best_colors}

