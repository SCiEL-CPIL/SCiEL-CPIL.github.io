import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def is_valid(n, colors):
        """
        Check if the coloring is valid (no C4 in any color class).
        colors is a list of length n*(n-1)/2
        """
        # Build adjacency lists for each color
        adj = [set() for _ in range(n)]
        # Actually, we need to check for each color separately.
        # It's faster to build adjacency bitmasks for each color.
        # Since n <= 29, we can use Python integers as bitsets.
        
        # Map edge index to (i, j)
        # colors[i, j] is stored in a flat array.
        # Index k corresponds to i < j.
        # We can reconstruct i, j from k or store the mapping.
        
        # Better: build adjacency bitmasks for each color.
        adj_masks = [ [0] * n for _ in range(5) ]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_masks[c][i] |= (1 << j)
                adj_masks[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Count common neighbors
                    common = adj_masks[c][u] & adj_masks[c][v]
                    if bin(common).count("1") > 1:
                        return False
        return True

    def get_seed_n11():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 0: d = n
                # distances 1 to 5. 
                # min(d, n-d) gives 1..5 for n=11?
                # 1,2,3,4,5. Yes.
                # But wait, j-i is always positive in the loop i<j.
                # d = j-i. Range 1..10.
                # min(d, 11-d) -> 1,2,3,4,5,4,3,2,1.
                # This maps distance 6 to color 4 (min(6,5)=5 -> 5-1=4).
                # Distance 5 maps to color 4 (min(5,6)=5 -> 4).
                # So distance 5 and 6 are both color 4.
                # This is the cyclic decomposition into 5 cycles?
                # No, for n=11, the cycle of distance k is a Hamiltonian cycle.
                # The set of edges at distance 5 is a 5-cycle? No, in Z11, distance 5 is also a Hamiltonian cycle (step 5 is coprime to 11).
                # So we have 5 Hamiltonian cycles.
                # Each color class is a union of cycles?
                # Wait, the seed code says: min(j - i, n - j + i) - 1.
                # For n=11, j-i in 1..10.
                # min(d, 11-d).
                # d=1 -> 1 -> color 0
                # d=2 -> 2 -> color 1
                # d=3 -> 3 -> color 2
                # d=4 -> 4 -> color 3
                # d=5 -> 5 -> color 4
                # d=6 -> 5 -> color 4
                # d=7 -> 4 -> color 3
                # d=8 -> 3 -> color 2
                # d=9 -> 2 -> color 1
                # d=10 -> 1 -> color 0
                # So color 0 has edges of distance 1 and 10. This is the Hamiltonian cycle 0-1-2...
                # Color 1 has distance 2 and 9.
                # Color 2 has distance 3 and 8.
                # Color 3 has distance 4 and 7.
                # Color 4 has distance 5 and 6.
                # Since each color class is a single Hamiltonian cycle (union of two paths that form one cycle? No, distance 1 and 10 form the cycle 0-1-...-10-0. It is a single cycle).
                # A single cycle $C_{11}$ contains no $C_4$.
                # So this is valid.
                colors.append(min(j - i, n - (j - i)) - 1)
        return n, colors

    def local_search(n, time_budget):
        """
        Attempt to find a valid coloring for n vertices.
        Returns colors list if found, None otherwise.
        """
        if time.time() - start_time > time_limit:
            return None

        # Initialize with random colors
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # We need a fast way to evaluate and update.
        # Let's build adj masks.
        # adj[c][i] is a bitmask of neighbors of i in color c.
        adj = [ [0] * n for _ in range(5) ]
        
        def build_adj(colors):
            for c in range(5):
                for i in range(n):
                    adj[c][i] = 0
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
        
        def get_badness():
            # Sum of max(0, common-1) for all pairs and colors
            bad = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[c][u] & adj[c][v]
                        cnt = bin(common).count("1")
                        if cnt > 1:
                            bad += cnt - 1
            return bad

        def calc_badness_for_edge_change(u, v, c_old, c_new):
            """
            Estimate change in badness if we change edge (u,v) from c_old to c_new.
            This is complex to do incrementally without updating the whole state.
            For simplicity in the first pass, we will just recalculate locally affected pairs.
            Pairs affected are those involving u or v.
            """
            # This function is a placeholder. We will implement a simpler "try and check" loop
            # if n is small, but for n up to 20, we need efficiency.
            # Let's use a different strategy:
            # Pick a random edge.
            # Try all 5 colors.
            # For each color, check if the resulting graph is valid? No, that's O(n^2) per check.
            
            # Let's stick to a simple hill climbing:
            # 1. Calculate total badness.
            # 2. If 0, return.
            # 3. Pick a random edge (u,v).
            # 4. For each color c in 0..4:
            #    Temporarily change color.
            #    Update adj masks.
            #    Calculate new badness? 
            #    Updating badness efficiently is hard.
            
            # Alternative: 
            # Use the fact that n is small.
            # Just check validity of the whole graph after a change?
            # O(n^2 * 5) per check is too slow if we do many iterations.
            
            # Let's try a different initialization: 
            # If n is small (<= 13), we can try random restarts.
            # For larger n, it's very hard.
            
            pass

        # Given the difficulty of efficient local search for this specific constraint, 
        # and the time limit, let's focus on finding n=12 or 13 if possible, 
        # but primarily ensure n=11 is returned.
        
        # Let's try a simple greedy random search for n=12, 13.
        # For n=12, num_edges = 66.
        # Random coloring has expected badness.
        
        for attempt in range(20): # Max 20 attempts per n
            if time.time() - start_time > time_limit:
                return None
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            # Simple local search
            for step in range(10000):
                if time.time() - start_time > time_limit:
                    return None
                # Check if valid
                # This check is O(n^2 * 5). For n=13, ~850 ops. 10000 steps -> 8.5M ops. Feasible.
                if is_valid(n, colors):
                    return colors
                
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                # Find u, v
                # We can precompute edge indices to (u,v)
                # Or just pick random u < v
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Find index of (u,v)
                # Index formula: 
                # i-th vertex has n-1-i edges to the right?
                # No, standard lexicographic order.
                # Count edges before (u,v).
                # Edges starting at 0: n-1
                # Edges starting at 1: n-2
                # ...
                # Edges starting at u-1: n-(u-1)-1 = n-u
                # Sum = (n-1) + ... + (n-u)
                # This is sum_{k=0}^{u-1} (n-1-k)
                # = u*(n-1) - u*(u-1)/2
                # Then within u, v is the (v-u)-th edge?
                # Edges from u are to u+1, u+2...
                # v-u is the offset.
                idx = u * (2 * n - u - 1) // 2 + (v - u - 1)
                
                # Try changing color
                old_c = colors[idx]
                new_c = (old_c + 1) % 5
                colors[idx] = new_c
        
        return None

    # Try to find larger n
    best_n = 11
    best_colors = None
    
    # Get seed
    n_seed, colors_seed = get_seed_n11()
    if is_valid(n_seed, colors_seed):
        best_n = n_seed
        best_colors = colors_seed
    
    # Try to improve
    for n in range(12, 26):
        if time.time() - start_time > time_limit - 10:
            break
        # Try local search
        res = local_search(n, 5.0)
        if res:
            best_n = n
            best_colors = res
            # If we found a valid one, we can try to go higher immediately
            # But local_search is slow. 
            # Let's just break and try to see if we can find n+1 quickly?
            # Or continue?
            # If we found n, we can try n+1.
            # But the search might be expensive.
            # Let's just return the best found so far to be safe?
            # No, we want to maximize.
            # Let's try one step ahead.
            pass
        else:
            # If we failed for n, it's likely we fail for n+1
            # But sometimes n=13 works and n=12 failed? Unlikely.
            # Let's stop if we fail 2 in a row?
            break

    return {'n': best_n, 'colors': best_colors}
