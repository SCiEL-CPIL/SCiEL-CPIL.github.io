import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    # --- Exact triangle-free check using bitmasks ---
    def is_triangle_free_masks(adj_masks, n):
        for i in range(n):
            # For each pair of neighbors j < k of i, check if j-k is an edge
            neighbors = adj_masks[i] & ~((1 << (i + 1)) - 1)  # neighbors > i
            # Actually simpler: for each i, for each j in adj[i] with j>i, check common neighbors
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    common = adj_masks[i] & adj_masks[j] & ~((1 << (j + 1)) - 1)
                    if common:
                        return False
        return True

    # --- Exact independent set check: returns True if alpha >= k ---
    def has_independent_set_masks(adj_masks, n, k):
        """Returns True if there exists an independent set of size k."""
        # Backtracking with bitmask
        # State: candidates mask, current count
        # Branch on vertex with max degree in induced subgraph
        
        def dfs(candidates, count):
            if count == k:
                return True
            if candidates.bit_count() < k - count:
                return False
            
            # Pick vertex with max degree in induced subgraph
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return False
            
            v_bit = 1 << best_v
            
            # Include best_v: remove it and its neighbors
            new_cand = candidates & ~v_bit & ~adj_masks[best_v]
            if dfs(new_cand, count + 1):
                return True
            
            # Exclude best_v
            new_cand = candidates & ~v_bit
            if dfs(new_cand, count):
                return True
            
            return False
        
        full = (1 << n) - 1
        return dfs(full, 0)

    def count_triangles_masks(adj_masks, n):
        """Count triangles in graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    common = adj_masks[i] & adj_masks[j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
        return count

    def get_max_indep_lower_bound_masks(adj_masks, n):
        """Greedy lower bound on alpha: pick vertex with min degree, remove it and neighbors."""
        candidates = (1 << n) - 1
        count = 0
        while candidates:
            # Pick vertex with min degree in induced subgraph
            best_v = -1
            min_deg = n
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            count += 1
            candidates &= ~(1 << best_v) & ~adj_masks[best_v]
        return count

    def get_edges_from_mask(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # --- Paley graph construction on q vertices (q prime, q ≡ 1 mod 4) ---
    def is_quadratic_residue(a, p):
        if a == 0:
            return False
        return pow(a, p // 2, p) == 1

    def paley_graph(q):
        """Construct Paley graph on q vertices. q must be prime ≡ 1 mod 4."""
        adj_masks = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                d = (j - i) % q
                if is_quadratic_residue(d, q):
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    # --- Simulated Annealing ---
    def simulated_annealing(adj_masks_init, n, target_k=10, max_time=120):
        """
        SA to find triangle-free graph with alpha < target_k.
        Energy = triangle_count * 100 + max(0, alpha_lower_bound - (target_k-1)) * 10
        Moves: flip a random edge (add or remove).
        Constraint: never create a triangle (reject moves that add triangle).
        """
        adj_masks = list(adj_masks_init)
        current_triangles = count_triangles_masks(adj_masks, n)
        
        # Initial alpha estimate
        current_alpha_lb = get_max_indep_lower_bound_masks(adj_masks, n)
        
        best_adj = list(adj_masks)
        best_triangles = current_triangles
        best_alpha_lb = current_alpha_lb
        best_energy = current_triangles * 100 + max(0, current_alpha_lb - (target_k - 1)) * 10
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        steps_per_check = 500
        
        step = 0
        start_sa = time.time()
        
        while time_left() > 10:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            has_edge = (adj_masks[u] >> v) & 1
            
            if has_edge:
                # Remove edge: always safe (reduces or maintains triangle-free)
                new_adj = list(adj_masks)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                new_tri = count_triangles_masks(new_adj, n)
                # Removing an edge can only reduce triangles
                new_alpha_lb = get_max_indep_lower_bound_masks(new_adj, n)
                
                new_energy = new_tri * 100 + max(0, new_alpha_lb - (target_k - 1)) * 10
                delta_e = new_energy - (current_triangles * 100 + max(0, current_alpha_lb - (target_k - 1)) * 10)
                
                if delta_e <= 0 or random.random() < math.exp(-delta_e / max(T, T_min)):
                    adj_masks = new_adj
                    current_triangles = new_tri
                    current_alpha_lb = new_alpha_lb
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj = list(adj_masks)
                        best_triangles = current_triangles
                        best_alpha_lb = current_alpha_lb
            else:
                # Add edge: must not create triangle
                # Check if u and v share any common neighbor
                common = adj_masks[u] & adj_masks[v]
                if common:
                    continue  # Would create triangle, skip
                
                new_adj = list(adj_masks)
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                # Adding edge (u,v) creates triangles with each common neighbor of u and v
                # But we already checked common == 0, so no new triangles
                # However, adding edge might create triangles involving u-v and existing edges
                # Actually: triangle (u,v,w) exists iff w is common neighbor of u and v
                # Since common == 0, no new triangles are created
                new_tri = current_triangles  # unchanged
                new_alpha_lb = get_max_indep_lower_bound_masks(new_adj, n)
                
                new_energy = new_tri * 100 + max(0, new_alpha_lb - (target_k - 1)) * 10
                current_energy = current_triangles * 100 + max(0, current_alpha_lb - (target_k - 1)) * 10
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or random.random() < math.exp(-delta_e / max(T, T_min)):
                    adj_masks = new_adj
                    current_alpha_lb = new_alpha_lb
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj = list(adj_masks)
                        best_triangles = current_triangles
                        best_alpha_lb = current_alpha_lb
            
            step += 1
            if step % steps_per_check == 0:
                T = max(T * cooling, T_min)
        
        return best_adj, best_triangles, best_alpha_lb

    # --- Phase 1: Try Paley(41) and refine ---
    # Paley graph on 41 vertices
    # 41 is prime and 41 ≡ 1 mod 4, so Paley graph exists
    # Degree = (41-1)/2 = 20
    # It's strongly regular (41, 20, 9, 10)
    # Triangle count: each vertex is in 9*20/2... let's just compute
    # Actually SRG(41,20,9,10): triangles per vertex = 9, total = 41*9/3 = 123 triangles
    
    # This has too many triangles. We need to remove them.
    # But the structure is good: dense, regular, small alpha.
    
    # Alternative: Use a different approach.
    # The known result R(3,10) > 39 comes from specific constructions.
    # Let's try a different spectral approach: use a graph from a difference set or
    # construct via the "polar graph" of a symplectic space.
    
    # Actually, let's try a more direct approach:
    # Start with Paley(41), remove edges to eliminate triangles, then add edges back carefully.
    
    # Phase 1a: Paley(41) + triangle removal + SA
    if time_left() > 20:
        adj_41 = paley_graph(41)
        # Remove triangles greedily
        # Find all triangles and remove edges
        changed = True
        while changed and time_left() > 15:
            changed = False
            # Find a triangle
            found = False
            for i in range(41):
                if found:
                    break
                for j in range(i + 1, 41):
                    if found:
                        break
                    if (adj_41[i] >> j) & 1:
                        common = adj_41[i] & adj_41[j] & ~((1 << (j + 1)) - 1)
                        if common:
                            # Pick a vertex k from common
                            k = (common & -common).bit_length() - 1
                            # Remove edge (i,k) or (j,k) or (i,j)
                            # Remove the edge that's part of the most triangles? 
                            # Just remove (i, j) for simplicity
                            adj_41[i] &= ~(1 << j)
                            adj_41[j] &= ~(1 << i)
                            changed = True
                            found = True
                            break
        
        # Now check alpha
        tri_count = count_triangles_masks(adj_41, 41)
        alpha_lb = get_max_indep_lower_bound_masks(adj_41, 41)
        
        if tri_count == 0 and not has_independent_set_masks(adj_41, 41, 10):
            return {'n': 41, 'edges': get_edges_from_mask(adj_41, 41)}
        
        # Run SA to refine
        if time_left() > 30:
            best_adj, best_tri, best_alpha = simulated_annealing(
                adj_41, 41, target_k=10, max_time=min(time_left() - 20, 60)
            )
            if best_tri == 0 and not has_independent_set_masks(best_adj, 41, 10):
                return {'n': 41, 'edges': get_edges_from_mask(best_adj, 41)}
    
    # Phase 1b: Try n=40 with a different spectral construction
    # Use a subgraph of Paley(41) by removing one vertex
    if time_left() > 30:
        adj_41_full = paley_graph(41)
        # Remove vertex 40 to get n=40
        adj_40 = [0] * 40
        for i in range(40):
            adj_40[i] = adj_41_full[i] & ((1 << 40) - 1)  # mask out vertex 40
        
        # Remove triangles
        changed = True
        while changed and time_left() > 25:
            changed = False
            found = False
            for i in range(40):
                if found:
                    break
                for j in range(i + 1, 40):
                    if found:
                        break
                    if (adj_40[i] >> j) & 1:
                        common = adj_40[i] & adj_40[j] & ~((1 << (j + 1)) - 1)
                        if common:
                            k = (common & -common).bit_length() - 1
                            adj_40[i] &= ~(1 << j)
                            adj_40[j] &= ~(1 << i)
                            changed = True
                            found = True
                            break
        
        tri_count = count_triangles_masks(adj_40, 40)
        if tri_count == 0 and not has_independent_set_masks(adj_40, 40, 10):
            return {'n': 40, 'edges': get_edges_from_mask(adj_40, 40)}
        
        if time_left() > 30:
            best_adj, best_tri, best_alpha = simulated_annealing(
                adj_40, 40, target_k=10, max_time=min(time_left() - 20, 50)
            )
            if best_tri == 0 and not has_independent_set_masks(best_adj, 40, 10):
                return {'n': 40, 'edges': get_edges_from_mask(best_adj, 40)}
    
    # Phase 2: Multi-restart SA from different spectral seeds
    # Try several different structured starting points
    
    def random_spectral_start(n):
        """Create a random regular-ish graph with good spectral properties."""
        adj = [0] * n
        # Target degree ~ n/2
        target_deg = n // 2
        for i in range(n):
            # Pick random neighbors
            candidates = [j for j in range(n) if j != i]
            random.shuffle(candidates)
            for j in candidates[:target_deg]:
                if not ((adj[i] >> j) & 1):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    for attempt in range(3):
        if time_left() < 25:
            break
        
        for n_try in [40, 39]:
            if time_left() < 20:
                break
            
            adj_start = random_spectral_start(n_try)
            
            # Quick triangle removal
            changed = True
            while changed and time_left() > 15:
                changed = False
                found = False
                for i in range(n_try):
                    if found:
                        break
                    for j in range(i + 1, n_try):
                        if found:
                            break
                        if (adj_start[i] >> j) & 1:
                            common = adj_start[i] & adj_start[j] & ~((1 << (j + 1)) - 1)
                            if common:
                                adj_start[i] &= ~(1 << j)
                                adj_start[j] &= ~(1 << i)
                                changed = True
                                found = True
                                break
            
            tri_count = count_triangles_masks(adj_start, n_try)
            if tri_count == 0:
                if not has_independent_set_masks(adj_start, n_try, 10):
                    return {'n': n_try, 'edges': get_edges_from_mask(adj_start, n_try)}
            
            # Run SA
            if time_left() > 25:
                best_adj, best_tri, best_alpha = simulated_annealing(
                    adj_start, n_try, target_k=10, max_time=min(time_left() - 15, 30)
                )
                if best_tri == 0 and not has_independent_set_masks(best_adj, n_try, 10):
                    return {'n': n_try, 'edges': get_edges_from_mask(best_adj, n_try)}
    
    # Phase 3: Fallback - construct a known valid small graph
    # C_5 union C_5 union C_5 union C_5 = 20 vertices, alpha = 4, triangle-free
    # Or better: use a circulant graph
    # C_5 with chords: Petersen graph has 10 vertices, triangle-free, alpha=4
    # Disjoint union of 4 Petersen graphs: 40 vertices, alpha=16. Too big.
    
    # Let's use a circulant: vertices 0..n-1, edge (i,j) if (j-i) mod n in S
    # For n=39, try S = {1,2,3,4,5,6,7,8,9,10} (degree 20)
    # This is a circulant, likely has small alpha if S is large enough
    
    # Actually, the simplest reliable fallback:
    # Complete bipartite K_{19,19} is triangle-free but alpha=19. No good.
    # 
    # Let's just return a small valid graph as ultimate fallback
    # C_5: n=5, triangle-free, alpha=2
    # Better: C_5 x C_5 (Cartesian product) = 25 vertices
    # Or just multiple C_5's: 8 * C_5 = 40 vertices, alpha = 8*2 = 16. No.
    
    # Single C_5: n=5
    # To get larger: use the Clebsch graph (16 vertices, triangle-free, alpha=5)
    # Or the Hoffman-Singleton graph (50 vertices, triangle-free, alpha=... too big)
    
    # Let's construct the Clebsch graph: 16 vertices, triangle-free, alpha=5
    # Vertices: all binary strings of length 4 with even weight (16 of them)
    # Two vertices adjacent if Hamming distance = 2
    
    def clebsch_graph():
        n = 16
        # Vertices: 0..15, where i is a 4-bit number with even popcount
        # Actually, let's use all 16 vertices 0..15
        # Adjacent if popcount(i XOR j) == 2
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = bin(i ^ j).count('1')
                if d == 2:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n
    
    adj_c, n_c = clebsch_graph()
    # Check: triangle-free? The Clebsch graph is known to be triangle-free.
    # Alpha = 5. So it's valid for k=10.
    # n=16.
    
    # Can we do better? Let's try to construct a valid graph on more vertices.
    # Disjoint union of two Clebsch graphs: 32 vertices, alpha = 10. NOT valid (need alpha < 10).
    # Disjoint union of one Clebsch: 16 vertices, alpha = 5. Valid.
    
    # What about a different construction?
    # The Andrasfai graph A_5: 5 vertices (C_5). A_9: 25 vertices? 
    # Actually Andrasfai graphs A_n for odd n >= 5: 
    # A_5 = C_5 (5 vertices)
    # A_7: 7 vertices? No, A_n has n vertices for the basic construction.
    # Wait, the Andrasfai graph A_n is defined on n vertices for odd n >= 5.
    # A_5 = C_5, A_7 = C_7? No, C_7 has alpha=3, triangle-free. Valid!
    # A_9: 9-vertex graph? The standard Andrasfai graph A_9 has 9 vertices? 
    # Actually I think the Andrasfai graph A_n for n odd >= 5 is just C_n for n=5,7
    # and for n>=9 it's a specific triangle-free graph with alpha = (n-1)/2.
    # For n=17: alpha = 8 < 10. Triangle-free. Valid!
    # For n=19: alpha = 9 < 10. Triangle-free. Valid!
    # For n=21: alpha = 10. NOT valid.
    
    # So a C_19 (cycle on 19 vertices) is triangle-free with alpha = 9 < 10. Valid!
    # But we can do much better.
    
    # Let's try: can we construct a valid graph on 30+ vertices?
    # The key constraint is alpha < 10, so alpha <= 9.
    # For a d-regular triangle-free graph on n vertices, Hoffman bound:
    # alpha <= n * (-lambda_2) / (d - lambda_2)
    # where lambda_2 is the second-smallest eigenvalue.
    
    # For our purposes, let's just return the best we can construct quickly.
    # C_19 is valid: n=19, alpha=9.
    
    # Can we get n=20? A 20-vertex triangle-free graph with alpha <= 9?
    # C_19 + isolated vertex: alpha = 10. No.
    # C_19 with one vertex connected: still might have alpha 9 or 10.
    
    # Let's just return C_19 as fallback, or try one more thing.
    
    # Actually, let me try a quick SA for n=35 with a good seed
    if time_left() > 15:
        # Start with a random 3-regular-ish graph
        n_fb = 35
        adj_fb = [0] * n_fb
        for i in range(n_fb):
            for j in range(i + 1, n_fb):
                if random.random() < 0.3:
                    adj_fb[i] |= (1 << j)
                    adj_fb[j] |= (1 << i)
        
        # Remove triangles
        changed = True
        while changed and time_left() > 10:
            changed = False
            found = False
            for i in range(n_fb):
                if found:
                    break
                for j in range(i + 1, n_fb):
                    if found:
                        break
                    if (adj_fb[i] >> j) & 1:
                        common = adj_fb[i] & adj_fb[j] & ~((1 << (j + 1)) - 1)
                        if common:
                            adj_fb[i] &= ~(1 << j)
                            adj_fb[j] &= ~(1 << i)
                            changed = True
                            found = True
                            break
        
        if count_triangles_masks(adj_fb, n_fb) == 0:
            if not has_independent_set_masks(adj_fb, n_fb, 10):
                return {'n': n_fb, 'edges': get_edges_from_mask(adj_fb, n_fb)}
    
    # Ultimate fallback: C_19
    n_fb = 19
    edges_fb = []
    for i in range(n_fb):
        edges_fb.append([i, (i + 1) % n_fb])
    
    return {'n': n_fb, 'edges': edges_fb}
