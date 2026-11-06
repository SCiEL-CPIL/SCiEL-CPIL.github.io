import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 11
    best_colors = None
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Try to find larger n using Backtracking with Restarting
    # The key insight: A C4 is formed by two vertices sharing two common neighbors in the same color.
    # When assigning an edge (u,v) color c, we must ensure that u and v do not already share 
    # two common neighbors in color c.
    
    def solve_n(n, time_limit):
        """
        Attempts to find a valid 5-coloring of K_n.
        Returns colors list or None.
        """
        if time.time() > time_limit:
            return None
            
        num_edges = n * (n - 1) // 2
        # Precompute edge indices
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j))
        
        # We will use a recursive backtracking solver.
        # To avoid recursion depth issues, we might need to increase limit or use iterative.
        # n=16 -> 120 edges. Recursion depth 120 is fine.
        
        # State:
        # colors: list of length num_edges, -1 if unassigned
        # adj_count[c][u][v] is not needed, we can just maintain adj lists or counts.
        # To check if assigning (u,v)=c creates a C4:
        # We need to know how many common neighbors w exist such that (u,w)=c and (w,v)=c.
        # Let's maintain for each color c, a 2D array common_count_c[u][v] = number of w such that (u,w)=c and (w,v)=c.
        # When we assign (u,v)=c:
        #   1. Check if common_count_c[u][v] >= 2. If so, invalid.
        #   2. Update common_count for all w.
        #      For each w != u, v:
        #         if (u,w) is c and (w,v) is c, then common_count_c[u][v] increases?
        #         No, the check is done *before* assignment.
        #         After assignment, we must update common_count_c[x][y] for all pairs (x,y) that are "affected".
        #         The edge (u,v) being c affects common_count_c[u][w] and common_count_c[v][w] for all w.
        #         Specifically, if (u,w) is c, then u and w now share v as a common neighbor in color c.
        #         So common_count_c[u][w] += 1.
        #         Similarly common_count_c[v][w] += 1.
        
        colors = [-1] * num_edges
        # common_count[c][u][v]
        # Use a 3D list or 5 2D lists.
        cc = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(5)]
        
        # To speed up, we can precompute which edges are adjacent to which.
        # But n is small.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
                
            if idx == num_edges:
                return True
                
            u, v = edge_list[idx]
            
            # Try colors 0..4
            # Heuristic: Try colors that are less constrained?
            # For now, randomize order to allow restarts to work differently.
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check constraint
                if cc[c][u][v] >= 2:
                    continue
                
                # Assign
                colors[idx] = c
                
                # Update common counts
                # The edge (u,v) is now color c.
                # For any w, if (u,w) is c, then u and w share v.
                # So cc[c][u][w] += 1 and cc[c][w][u] += 1
                # Similarly for v.
                
                # We need to know the color of (u,w) and (v,w).
                # This requires looking up colors of other edges.
                # To make this O(1) or fast, we should maintain adjacency info.
                # Let's maintain adj_color[c][u] as a set or bitmask of neighbors.
                # But we are already using cc.
                # Let's just iterate w.
                
                updated = []
                valid = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Check (u, w)
                    # Find index of (u, w)
                    # We need a fast lookup for edge index.
                    # Precompute edge_index[(u,w)]
                    
                    # This is getting slow if we do it naively in Python inside recursion.
                    # Let's precompute edge_index.
                    pass
                
                # This approach is too slow in Python for n=16 with 120 edges and deep recursion if we iterate all w for every assignment.
                # We need a faster constraint check.
                
                # Alternative: Use the bitmask approach from the seed but with backtracking?
                # Or, simpler: Just use the seed's local search but with a MUCH better initialization.
                
                # Let's abort this specific backtracking implementation idea if it's too complex to optimize in time.
                # Let's go back to Local Search but with a **Deterministic Algebraic Init** for n=16.
                
                # What is a good algebraic init for n=16?
                # We established that slope coloring creates K4s which have C4s.
                # However, if we **break** the K4s by permuting vertices?
                # Or, use a different structure.
                
                # Let's try a **Random Init + Fast Local Search** but with **Multiple Restarts** and **Better Cooling**.
                # The seed's failure might be due to insufficient restarts or bad cooling.
                
                # Let's try a **Genetic Algorithm** with a simple fitness function.
                # Population of size 20.
                # Crossover: Uniform.
                # Mutation: Change 1-5 edges.
                # Selection: Tournament.
                # Fitness: -violations.
                
                # This is structurally different from SA.
                
                return False # Placeholder to exit this function structure
                
        # Since the backtracking implementation is getting complex and potentially slow, 
        # let's pivot to a **Robust Local Search with Multiple Restarts and Diversification**.
        # The key change: Use **Bitmask** for fast violation counting (as in seed) but 
        # add **Iterated Local Search (ILS)**: 
        # 1. Local Search to local optimum.
        # 2. Perturb (change k random edges).
        # 3. Local Search again.
        # This explores the landscape better than simple SA.
        
        return None

    # Implement Iterated Local Search (ILS)
    
    def count_violations_fast(colors, n):
        """
        Fast violation count using bitmasks.
        Returns number of C4s (or excess common neighbors).
        """
        violations = 0
        # adj_mask[c][u]
        adj_mask = [ [0] * n for _ in range(5) ]
        
        # Build masks
        # We need to map (i,j) to index.
        # For speed, we can precompute the list of (i,j) for the given n.
        # But this function is called many times.
        # Let's pass edge_list or compute index on fly?
        # Computing index on fly is slow.
        # Let's precompute edge_list for this n.
        
        # To avoid passing edge_list, let's assume standard lexicographic order and use a helper.
        # Or, better, build the masks in the main loop where we have the structure.
        
        # Let's rewrite the main loop to be self-contained with precomputed structures.
        pass

    # Let's implement the ILS loop properly.
    
    for n in range(12, 22): # Try up to 21
        if time.time() > deadline - 10.0:
            break
            
        num_edges = n * (n - 1) // 2
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_idx = { (i,j): k for k, (i,j) in enumerate(edge_list) }
        
        def get_violations(colors):
            # Build adjacency masks
            adj_mask = [ [0] * n for _ in range(5) ]
            for k, (i, j) in enumerate(edge_list):
                c = colors[k]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                
            violations = 0
            for c in range(5):
                for u in range(n):
                    mask_u = adj_mask[c][u]
                    if mask_u == 0: continue
                    # Only check v > u to avoid double counting?
                    # No, common neighbors is symmetric.
                    # We just need to count pairs (u,v) with common neighbors > 1.
                    # Actually, we can just sum (count - 1) for all pairs.
                    for v in range(u + 1, n):
                        mask_v = adj_mask[c][v]
                        if mask_v == 0: continue
                        common = mask_u & mask_v
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
            return violations

        def local_search(colors, max_iters, time_limit):
            current_violations = get_violations(colors)
            if current_violations == 0:
                return colors, 0
                
            best_colors = colors[:]
            best_violations = current_violations
            temp = 1.0
            cooling = 0.9995
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                    
                if current_violations == 0:
                    break
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                new_violations = get_violations(colors)
                delta = new_violations - current_violations
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                else:
                    colors[e_idx] = old_color
                    
                temp *= cooling
                
            return best_colors, best_violations

        # ILS Loop
        found = False
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random Init
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Local Search
            colors, viol = local_search(colors, 5000, deadline - 5.0)
            
            if viol == 0:
                best_n = n
                best_colors = colors
                found = True
                break
            else:
                # Perturb
                # Change 5% of edges
                num_changes = max(1, num_edges // 20)
                for _ in range(num_changes):
                    e_idx = random.randint(0, num_edges - 1)
                    colors[e_idx] = random.randint(0, 4)
                    
                # Local Search on perturbed
                colors, viol = local_search(colors, 5000, deadline - 5.0)
                if viol == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break

        if found:
            # Try to increase n?
            # If we found n, we can try n+1 next iteration.
            # But if we fail n+1, we should return n.
            # The loop structure handles this: if n=16 works, we try 17.
            # If 17 fails, we break and return 16.
            # However, if 17 is hard, we might waste time.
            # Let's add a check: if we failed n, break?
            # No, random init might just be bad.
            # But if n=16 is found, n=17 is likely to be found with more restarts.
            # Let's continue.
            pass
        else:
            # If we couldn't find n, likely can't find n+1.
            # But maybe n was just unlucky?
            # Let's break to be safe and return best_n.
            # Actually, if n=12 fails, n=13 is harder.
            # If n=16 fails, n=17 is harder.
            # So breaking is a good heuristic.
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
