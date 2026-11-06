import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    # Cycle i uses edges (j, j+i) mod 11 for j=0..10. 
    # For n=11, distance d in 1..5.
    # Edge (i, j) with i<j. dist = min(j-i, 11-(j-i)).
    # Color = dist - 1.
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Strategy: Try to find valid colorings for n > 11 using Greedy Construction + Repair.
    # The constraint "No C4 in any color class" implies that for any color c, 
    # the graph G_c is C4-free.
    # A strong necessary condition: For any color c, the degree of any vertex in G_c 
    # is not strictly limited by C4-freeness alone (e.g., stars are C4-free), 
    # BUT if two vertices u, v share two neighbors x, y in G_c, we have a C4.
    # So, for any pair (u, v), the number of common neighbors in G_c must be <= 1.
    
    # Greedy Construction Idea:
    # Add vertices 0, 1, ..., n-1 one by one.
    # When adding vertex k, assign colors to edges (k, j) for j < k.
    # Constraint: For each color c, the set of neighbors of k in G_c, let's call N_c(k),
    # must not create a C4.
    # A C4 is formed if there exist j1, j2 in N_c(k) such that j1 and j2 share a common neighbor in G_c.
    # Actually, simpler: If we maintain the property that for all existing vertices u, v, 
    # common_neighbors_c(u, v) <= 1, adding k requires that for any c,
    # the new neighbors of k in G_c don't cause any pair (u, v) to have >1 common neighbor.
    # This is complex to check incrementally.
    
    # Alternative: Random initialization + Fast Local Search with "Conflict Repair".
    # The previous program used SA. Let's use a more directed approach:
    # 1. Initialize with a structured pattern (e.g., based on finite geometry or random).
    # 2. Identify conflicts: Pairs (u, v) with >1 common neighbor in color c.
    # 3. Resolve conflicts by changing colors of edges incident to u, v, or the common neighbors.
    
    # Let's try n from 12 up to a limit where time permits.
    # We know n=11 is valid. 
    # Is n=12 possible? R_5(C4) is likely > 11. 
    # Let's attempt to find a valid coloring for n=12, 13, etc.
    
    def get_idx(i, j, n):
        # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
        # Index = i*(n-1) - i*(i+1)//2 + j - i - 1
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def check_validity(colors, n):
        # Fast check using bitmasks
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    def count_violations(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
        return violations

    def repair_local_search(colors, n, max_time):
        """
        Local search to eliminate C4s.
        Focus on vertices/edges involved in conflicts.
        """
        current_violations = count_violations(colors, n)
        if current_violations == 0:
            return colors, True
        
        # Precompute edge map for fast access
        edge_map = {}
        inv_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                inv_map[idx] = (i, j)
                idx += 1
        
        num_edges = n * (n - 1) // 2
        
        # Simulated Annealing with aggressive cooling and conflict-focused moves
        temp = 5.0
        cooling = 0.99
        min_temp = 0.01
        
        # Track best state
        best_colors = colors[:]
        best_violations = current_violations
        
        while temp > min_temp and time.time() < max_time:
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            i, j = inv_map[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_violations = count_violations(colors, n)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color
            
            temp *= cooling
            
        # Final check
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Try to find solutions for n > 11
    # We will try a few restarts for each n.
    # If we find a valid one, we update best_n and best_colors.
    
    # To maximize n, we should try larger n first if we have time, 
    # but finding a valid coloring is harder for larger n.
    # Let's try 12, 13, 14, 15.
    
    # Heuristic: Use a structured initial guess for better convergence.
    # For n=12, maybe a random start is fine.
    
    for n in range(12, 16):
        if time.time() > deadline - 20.0:
            break
        
        success = False
        num_restarts = 5
        time_per_restart = (deadline - 20.0 - time.time()) / (num_restarts * (16 - n)) # rough estimate
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize
            # Strategy: Start with random, but maybe bias towards balanced degrees?
            # Just random is okay if SA is strong.
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Run local search
            # Allocate time based on remaining budget
            t_limit = time.time() + max(1.0, (deadline - 10.0 - time.time()) / 2.0)
            result_colors, is_valid = repair_local_search(colors, n, t_limit)
            
            if is_valid:
                best_n = n
                best_colors = result_colors
                success = True
                break # Move to next n
        
        # If we failed for n, it might be hard. 
        # Should we try n+1? Probably not if n is hard. 
        # But if n=12 is easy, n=13 might be possible.
        # If n=12 fails, n=13 is very likely to fail in this time.
        # So break if we fail a small n? 
        # Actually, sometimes a random seed works for 13 but not 12? Unlikely.
        # Let's break if we fail n=12 to save time, as 12 is the minimum improvement.
        # However, if we have lots of time, we could try more restarts.
        # Given 150s limit, and 12 being the target, if we fail 12 quickly, we return 11.
        # If we find 12, we try 13.
        
        if not success:
            # If we couldn't find 12, 13 is unlikely to be found with this method in remaining time.
            # But let's check time.
            if time.time() < deadline - 30.0 and n == 12:
                # Maybe try more restarts for 12?
                # The loop above already did 5 restarts.
                # Let's just break.
                break
            else:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
