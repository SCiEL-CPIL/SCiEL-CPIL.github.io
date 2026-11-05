"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Construct a C4-free graph with maximum n and minimum degree >= n-39.
    Uses bitsets for fast C4 checking and a greedy + local search approach.
    """
    
    def verify_c4_free(adj_bits, n):
        """Check if graph is C4-free using bitsets."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    continue
                # Check for C4: u-v-x-y-u
                # x in N(v) \ {u}, y in N(u) \ {v}
                # If any x in N(v)\{u} is connected to any y in N(u)\{v}, C4 exists.
                
                sv = adj_bits[v] & ~(1 << u)
                su = adj_bits[u] & ~(1 << v)
                
                # Check if any node in sv is connected to any node in su
                temp_sv = sv
                while temp_sv:
                    lsb = temp_sv & -temp_sv
                    x = lsb.bit_length() - 1
                    if (adj_bits[x] & su):
                        return False
                    temp_sv ^= lsb
        return True

    def try_build(n, seed, min_deg_target):
        random.seed(seed)
        adj_bits = [0] * n
        degree = [0] * n
        # dist2_bits[u] contains all vertices at distance exactly 2 from u (via one intermediate node)
        # This is used for fast C4 checking during greedy addition.
        dist2_bits = [0] * n

        def get_neighbors(u, exclude=-1):
            """Yield neighbors of u, optionally excluding one."""
            mask = adj_bits[u]
            if exclude >= 0:
                mask &= ~(1 << exclude)
            while mask:
                lsb = mask & -mask
                yield lsb.bit_length() - 1
                mask ^= lsb

        def can_add_edge_slow(u, v):
            """Check if adding edge (u,v) creates a C4. O(d) check."""
            sv = adj_bits[v] & ~(1 << u)
            su = adj_bits[u] & ~(1 << v)
            
            if sv == 0 or su == 0:
                return True
                
            # Iterate over the smaller set
            count_sv = bin(sv).count('1')
            count_su = bin(su).count('1')
            
            if count_sv < count_su:
                temp = sv
                target = su
            else:
                temp = su
                target = sv
                
            while temp:
                lsb = temp & -temp
                node = lsb.bit_length() - 1
                if adj_bits[node] & target:
                    return False
                temp ^= lsb
            return True

        def can_add_edge_fast(u, v):
            """Check if adding edge (u,v) creates a C4 using precomputed dist2. O(1) check."""
            if (adj_bits[u] >> v) & 1:
                return False
            # If v is in dist2_bits[u], there is a path u-x-y-v, so adding (u,v) creates a C4.
            if dist2_bits[u] & (1 << v):
                return False
            if dist2_bits[v] & (1 << u):
                return False
            return True

        def update_dist2_add(u, v):
            """Update dist2_bits when adding edge (u, v)."""
            nu = adj_bits[u]
            nv = adj_bits[v]
            
            # For all x in N(u), N(v) becomes distance 2 from x
            temp_nu = nu
            while temp_nu:
                lsb = temp_nu & -temp_nu
                x = lsb.bit_length() - 1
                dist2_bits[x] |= nv
                temp_nu ^= lsb
                
            # For all y in N(v), N(u) becomes distance 2 from y
            temp_nv = nv
            while temp_nv:
                lsb = temp_nv & -temp_nv
                y = lsb.bit_length() - 1
                dist2_bits[y] |= nu
                temp_nv ^= lsb
                
            # u's dist2 gains N(v)
            dist2_bits[u] |= nv
            # v's dist2 gains N(u)
            dist2_bits[v] |= nu

        def recalc_dist2():
            """Recalculate dist2_bits from scratch. Used after swaps."""
            for i in range(n):
                dist2_bits[i] = 0
            for u in range(n):
                nu = adj_bits[u]
                temp_nu = nu
                while temp_nu:
                    lsb = temp_nu & -temp_nu
                    x = lsb.bit_length() - 1
                    dist2_bits[u] |= adj_bits[x]
                    temp_nu ^= lsb
            # Remove self-loops and direct neighbors from dist2? 
            # Actually, dist2 should be vertices at distance exactly 2.
            # The definition used in can_add_edge_fast is: 
            # if v in dist2[u], then there is x s.t. (u,x) and (x,v) are edges.
            # This creates a path u-x-v. Adding (u,v) creates a triangle? No, u-x-v-u is a C3?
            # Wait. C4 is u-v-x-y-u.
            # If we add (u,v), and there is a path u-x-y-v of length 3, we get a C4.
            # My dist2 definition above is distance 2 (path length 2).
            # Path u-x-v. Adding (u,v) creates a triangle u-v-x-u.
            # Triangles are allowed in C4-free graphs!
            # So `dist2_bits` as "distance 2" is WRONG for C4 checking.
            # We need "distance 3" or "neighbors of neighbors of neighbors"?
            # No.
            # C4: u-v-a-b-u.
            # If we add (u,v), we need to check if there is a path of length 3 from u to v.
            # i.e., exists a, b s.t. (u,a), (a,b), (b,v) are edges.
            # This is equivalent to: N(u) and N(v) have a common neighbor? No.
            # It is equivalent to: There exists a in N(u) and b in N(v) such that (a,b) is an edge.
            # This is the condition checked in `can_add_edge_slow`.
            #
            # Can we precompute this?
            # Let M be the adjacency matrix.
            # We want to know if (M^3)[u][v] > 0?
            # M^3 counts paths of length 3.
            # If we add (u,v), a C4 is formed if there is a path of length 3 between u and v.
            # So we need to check if the number of length-3 paths between u and v is > 0.
            #
            # Precomputing M^3 is O(n^3) or O(n^2 d) with bitsets.
            # M^3[u] = union_{x in N(u)} M^2[x]? No.
            # Let's stick to `can_add_edge_slow` for correctness, as the "fast" dist2 logic was flawed for C4 (it checks triangles).
            #
            # However, we can optimize `can_add_edge_slow` by using bitsets more effectively.
            pass

        # Phase 1: Greedy
        max_rounds = 200
        for rnd in range(max_rounds):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        continue
                    if can_add_edge_slow(u, v):
                        score = degree[u] + degree[v]
                        candidates.append((score, u, v))
            
            if not candidates:
                break
            
            candidates.sort()
            
            added = 0
            for score, u, v in candidates:
                if (adj_bits[u] >> v) & 1:
                    continue
                if can_add_edge_slow(u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    degree[u] += 1
                    degree[v] += 1
                    added += 1
                    if added >= 10:
                        break
            
            if added == 0:
                break
        
        # Phase 2: Local Search (Swaps)
        # Try to improve min degree by swapping edges
        for step in range(2000):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            low_verts = [i for i in range(n) if degree[i] < min_deg_target]
            if not low_verts:
                break
                
            improved = False
            random.shuffle(low_verts)
            
            for u in low_verts:
                # Try to find a non-neighbor z to connect to u
                # And a neighbor w to disconnect from u
                # This is a 1-move: Add (u,z), Remove (u,w)
                
                neighbors_u = list(get_neighbors(u))
                if not neighbors_u:
                    continue
                    
                non_neighbors_u = [v for v in range(n) if v != u and not ((adj_bits[u] >> v) & 1)]
                
                # Sort non-neighbors by degree (low first)
                non_neighbors_u.sort(key=lambda x: degree[x])
                # Sort neighbors by degree (high first, so we remove from high degree nodes)
                neighbors_u.sort(key=lambda x: -degree[x])
                
                for z in non_neighbors_u[:10]: # Increased from 5 to 10
                    if degree[z] >= min_deg_target:
                        continue
                        
                    # We need a swap: Remove (u, w), then Add (u, z).
                    # But removing (u, w) might make degree[w] < target.
                    # We only allow swap if degree[w] > min_deg_target.
                    
                    for w in neighbors_u[:10]: # Increased from 5 to 10
                        if degree[w] <= min_deg_target:
                            continue
                            
                        # Tentatively remove (u, w)
                        adj_bits[u] &= ~(1 << w)
                        adj_bits[w] &= ~(1 << u)
                        degree[u] -= 1
                        degree[w] -= 1
                        
                        # Try to add (u, z)
                        if can_add_edge_slow(u, z):
                            adj_bits[u] |= (1 << z)
                            adj_bits[z] |= (1 << u)
                            degree[u] += 1
                            degree[z] += 1
                            improved = True
                            break
                        else:
                            # Revert
                            adj_bits[u] |= (1 << w)
                            adj_bits[w] |= (1 << u)
                            degree[u] += 1
                            degree[w] += 1
                            
                    if improved:
                        break
                        
            if not improved:
                break

        # Verify
        min_deg = min(degree)
        if min_deg < min_deg_target:
            return None
        if not verify_c4_free(adj_bits, n):
            return None
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_target = n - 39
        if min_deg_target < 0:
            min_deg_target = 0
            
        start_time = time.time()
        seeds_tried = 0
        while time.time() - start_time < 110:
            result = try_build(n, seeds_tried, min_deg_target)
            if result is not None:
                return {'n': n, 'edges': result}
            seeds_tried += 1
            if seeds_tried > 200: # Increased seeds
                break
    
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
