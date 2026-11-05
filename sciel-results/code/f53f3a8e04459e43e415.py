"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Construct a C4-free graph with maximum n and minimum degree >= n-39.
    Uses an optimized greedy approach with local search.
    """
    
    def is_c4_free(adj, n):
        """
        Fast check if graph is C4-free.
        adj: list of integers (bitsets)
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # If there is any common neighbor w, and w is not u or v, 
                # wait. C4 is u-v-w-x-u. 
                # If u and v are adjacent, they cannot share a common neighbor w 
                # because that would form a triangle u-v-w-u. C4-free graphs 
                # can have triangles. 
                # A C4 is formed if u and v are NOT adjacent but share two neighbors?
                # No. A C4 is a cycle of length 4.
                # Edges (u,v), (v,w), (w,x), (x,u).
                # If we add edge (u,v), does it create a C4?
                # It creates a C4 if there exist x,y such that (u,x), (x,y), (y,v) are edges 
                # and x != v, y != u.
                # So we need to check if N(u) \ {v} and N(v) \ {u} have a common neighbor?
                # No.
                # Let N(u) be neighbors of u.
                # Adding (u,v) creates a C4 if there is a path of length 3 from u to v 
                # that doesn't use the edge (u,v).
                # Path: u - a - b - v.
                # So a in N(u)\{v}, b in N(v)\{u}, and (a,b) is an edge.
                
                # So for each edge (u,v), we check if there exists a in N(u)\{v} 
                # such that N(a) intersects N(v)\{u}.
                
                nu = adj[u] & ~(1 << v)
                nv = adj[v] & ~(1 << u)
                
                # Iterate over smaller set
                if bin(nu).count('1') < bin(nv).count('1'):
                    temp = nu
                    target = nv
                else:
                    temp = nv
                    target = nu
                
                while temp:
                    lsb = temp & -temp
                    a = lsb.bit_length() - 1
                    if adj[a] & target:
                        return False
                    temp ^= lsb
        return True

    def try_build(n, seed, min_deg_target):
        random.seed(seed)
        adj = [0] * n
        degree = [0] * n
        
        def can_add(u, v):
            """Check if adding edge (u,v) creates a C4."""
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            
            if nu == 0 or nv == 0:
                return True
            
            # Check intersection of N(a) and N(v)\{u} for a in N(u)\{v}
            # Or N(b) and N(u)\{v} for b in N(v)\{u}
            
            # Optimization: If deg(u) + deg(v) is small, it's fast.
            # We iterate over the smaller neighborhood.
            
            if bin(nu).count('1') < bin(nv).count('1'):
                temp = nu
                target = nv
            else:
                temp = nv
                target = nu
                
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if adj[a] & target:
                    return False
                temp ^= lsb
            return True

        # Phase 1: Greedy Construction
        # We want to maximize minimum degree.
        # Strategy: Repeatedly add the edge that increases the minimum degree the most,
        # or simply add edges between low-degree vertices if possible.
        
        # To speed up, we maintain a list of potential edges.
        # But n is small (<=47), so O(n^2) is fine.
        
        for _ in range(100): # Max iterations
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            # Find all valid edges
            valid_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if can_add(u, v):
                        # Score: prioritize edges involving low-degree nodes
                        score = degree[u] + degree[v]
                        valid_edges.append((score, u, v))
            
            if not valid_edges:
                break
            
            # Sort by score (ascending) to pick edges that help low-degree nodes first?
            # Or descending to build dense core?
            # For min-degree, picking edges between the two lowest degree nodes is usually good.
            valid_edges.sort(key=lambda x: x[0])
            
            added = False
            for score, u, v in valid_edges:
                # Re-check validity in case previous additions in this round invalidated it?
                # No, we only add one at a time in this loop structure if we break.
                # But we might want to add multiple.
                if can_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degree[u] += 1
                    degree[v] += 1
                    added = True
                    # Add a few more?
                    # Let's just add one and restart the loop to update min_deg.
                    break
            
            if not added:
                break

        # Phase 2: Local Search
        # Try to improve min degree by swapping edges.
        # Move: Remove (u,w), Add (u,z).
        # Conditions:
        # 1. (u,w) is an edge.
        # 2. (u,z) is not an edge.
        # 3. Adding (u,z) is C4-free.
        # 4. degree[w] > min_deg_target (so w doesn't drop below target)
        # 5. degree[u] < min_deg_target (so u improves)
        
        for step in range(5000):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            low_verts = [i for i in range(n) if degree[i] < min_deg_target]
            if not low_verts:
                break
                
            improved = False
            random.shuffle(low_verts)
            
            for u in low_verts:
                # Get neighbors and non-neighbors
                neighbors = []
                non_neighbors = []
                for v in range(n):
                    if v == u:
                        continue
                    if (adj[u] >> v) & 1:
                        neighbors.append(v)
                    else:
                        non_neighbors.append(v)
                
                # Sort neighbors by degree descending (remove from high degree)
                neighbors.sort(key=lambda x: -degree[x])
                # Sort non-neighbors by degree ascending (add to low degree? 
                # Actually, we want to add (u,z). z's degree increases.
                # We don't care as much about z's degree, just that it doesn't break C4.
                # But maybe adding to a low degree node is safer?
                # Let's just iterate.
                
                for z in non_neighbors:
                    if not can_add(u, z):
                        continue
                    
                    # If we can add (u,z) directly without removing anything, 
                    # why didn't Phase 1 do it?
                    # Phase 1 stops when no valid edges exist. 
                    # If can_add(u,z) is True, Phase 1 would have added it 
                    # unless it was not in valid_edges? 
                    # valid_edges included all pairs. 
                    # So if can_add is True now, it was True before?
                    # No, adding other edges might have made it invalid? 
                    # No, adding edges only adds constraints. 
                    # So if it's valid now, it was valid before.
                    # This implies Phase 1 failed to add it?
                    # Phase 1 adds edges one by one. 
                    # If we can add it now, we should just add it.
                    
                    # Wait, if we can add (u,z) without removing, we should just do it.
                    # Why is this in Local Search?
                    # Because Phase 1 might have terminated early due to "no valid edges" 
                    # but that check was global. 
                    # If can_add(u,z) is True, we add it.
                    
                    adj[u] |= (1 << z)
                    adj[z] |= (1 << u)
                    degree[u] += 1
                    degree[z] += 1
                    improved = True
                    break
                
                if improved:
                    continue # u is fixed, try next u? Or restart?
                    # Let's restart the step loop to recalculate min_deg.
                    # But we are in a step loop. 
                    # Breaking inner loops and continuing step loop is fine.
                    # Actually, if we improved, we should restart the search for low_verts?
                    # No, just continue to next u or break step.
                    # Let's break step and restart.
                    break
            
            if not improved:
                # Try swaps
                found_swap = False
                for u in low_verts:
                    neighbors = [v for v in range(n) if v != u and (adj[u] >> v) & 1]
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    
                    neighbors.sort(key=lambda x: -degree[x])
                    
                    for w in neighbors:
                        if degree[w] <= min_deg_target:
                            continue
                        
                        # Remove (u,w)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        degree[u] -= 1
                        degree[w] -= 1
                        
                        success = False
                        for z in non_neighbors:
                            if can_add(u, z):
                                adj[u] |= (1 << z)
                                adj[z] |= (1 << u)
                                degree[u] += 1
                                degree[z] += 1
                                success = True
                                break
                        
                        if success:
                            found_swap = True
                            break
                        else:
                            # Revert
                            adj[u] |= (1 << w)
                            adj[w] |= (1 << u)
                            degree[u] += 1
                            degree[w] += 1
                        
                    if found_swap:
                        break
                
                if not found_swap:
                    break

        # Verify
        min_deg = min(degree)
        if min_deg < min_deg_target:
            return None
        if not is_c4_free(adj, n):
            return None
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Try n=47 first
    for n in [47, 46, 45, 44, 43]:
        min_deg_target = n - 39
        if min_deg_target < 0:
            min_deg_target = 0
            
        start_time = time.time()
        seeds_tried = 0
        while time.time() - start_time < 115:
            result = try_build(n, seeds_tried, min_deg_target)
            if result is not None:
                return {'n': n, 'edges': result}
            seeds_tried += 1
            if seeds_tried > 200:
                break
    
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
