import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # Best known seed: Paley graph on 17 vertices
    # Paley(17) is K4-free and has independence number 4 (no independent 6-set)
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17
    
    # Build adjacency matrix for Paley(17)
    adj17 = np.zeros((17, 17), dtype=np.int8)
    for u, v in edges_17:
        adj17[u, v] = 1
        adj17[v, u] = 1

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def get_edges_from_masks(n, masks):
        edges = []
        for u in range(n):
            m = masks[u]
            # Only take v > u to avoid duplicates
            m &= ~((1 << (u + 1)) - 1)
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def count_k4(n, masks):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            count += 1
        return count

    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        count += 1
        return count

    def total_violations(n, masks):
        return count_k4(n, masks) + count_ind6(n, masks)

    def delta_k4(n, masks, u, v, adding):
        """
        Calculate change in K4 count when flipping edge (u, v).
        If adding=True, we add edge (u,v). We need to count how many new K4s are formed.
        A K4 containing (u,v) is formed by choosing 2 vertices {w,x} from N(u) cap N(v) such that w-x is an edge.
        If adding=False, we remove edge (u,v). We need to count how many K4s are destroyed.
        A K4 containing (u,v) exists if we choose 2 vertices {w,x} from N(u) cap N(v) such that w-x is an edge.
        Wait, if we remove (u,v), the K4s destroyed are exactly those that *used* (u,v).
        So in both cases, we look for pairs {w,x} in N(u) cap N(v) that are connected.
        However, if we are ADDING (u,v), the neighbors w,x must be neighbors of both u and v *before* the flip?
        No, N(u) and N(v) do not change when we flip (u,v) (since u is not in N(u) and v is not in N(v)).
        So N(u) cap N(v) is invariant.
        Thus, delta is the same magnitude, just sign depends on adding/removing.
        """
        common = masks[u] & masks[v]
        # We need to count edges among vertices in 'common'
        # But we must be careful: w and x must be distinct from u and v.
        # Since u is not in masks[u] and v is not in masks[v], u is not in common and v is not in common.
        # So common contains only vertices other than u and v.
        
        count = 0
        # Iterate over pairs in common
        m = common
        while m:
            lsb_w = m & -m
            wi = lsb_w.bit_length() - 1
            m ^= lsb_w
            # For each w, check neighbors in common that are greater than w
            x_mask = m & masks[wi]
            # x_mask contains neighbors of w that are in common and > w
            while x_mask:
                lsb_x = x_mask & -x_mask
                # xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                count += 1
        return count

    def delta_ind6(n, masks, u, v, adding):
        """
        Calculate change in Ind6 count when flipping edge (u, v).
        Independent set of 6 containing u,v requires 4 other vertices {w,x,y,z} that are:
        1. Not adjacent to u
        2. Not adjacent to v
        3. Mutually non-adjacent (forming an independent set of 4 in the subgraph induced by candidates)
        
        Candidates C = (Full \ {u,v}) \ (N(u) U N(v))
        If ADDING (u,v):
           Before: u and v were not connected.
           If we add (u,v), we might DESTROY independent sets that previously existed?
           No, an independent set cannot contain an edge.
           If we ADD an edge (u,v), any independent set containing both u and v is INVALID.
           So we REMOVE all independent 6-sets that contain u,v.
           The number removed is the number of independent 4-sets in C.
           
        If REMOVING (u,v):
           Before: u and v were connected. No independent set could contain both.
           If we REMOVE (u,v), we CREATE independent 6-sets that contain u,v.
           The number created is the number of independent 4-sets in C.
           
        So magnitude is the same: count independent 4-sets in C.
        """
        full_mask = (1 << n) - 1
        # Candidates are vertices not u, not v, not neighbor of u, not neighbor of v
        # C = Full & ~((1<<u) | (1<<v) | masks[u] | masks[v])
        C = full_mask & ~((1 << u) | (1 << v) | masks[u] | masks[v])
        
        # Count independent 4-sets in the subgraph induced by C.
        # This is equivalent to counting K4s in the complement of the subgraph induced by C.
        # Or simply, pick 4 vertices from C such that no edges exist between them in the original graph.
        
        count = 0
        m = C
        while m:
            lsb_w = m & -m
            wi = lsb_w.bit_length() - 1
            m ^= lsb_w
            
            # Candidates for x: in m, not adjacent to w
            # x_mask = m & ~masks[wi]
            x_mask = m & ~masks[wi]
            
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                
                # Candidates for y: in x_mask (so > x, not adj to w), not adjacent to x
                # y_mask = x_mask & ~masks[xi]
                y_mask = x_mask & ~masks[xi]
                
                while y_mask:
                    lsb_y = y_mask & -y_mask
                    yi = lsb_y.bit_length() - 1
                    y_mask ^= lsb_y
                    
                    # Candidates for z: in y_mask (so > y, not adj to w, not adj to x), not adjacent to y
                    # z_mask = y_mask & ~masks[yi]
                    z_mask = y_mask & ~masks[yi]
                    
                    # Count bits in z_mask
                    count += z_mask.bit_count()
                    
        return count

    def try_find_graph(n, time_budget, seed_adj=None):
        end_time = time.time() + time_budget
        
        # Initialize masks
        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        masks = build_masks(n, adj)
        current_violations = total_violations(n, masks)
        
        if current_violations == 0:
            return get_edges_from_masks(n, masks)
        
        best_violations = current_violations
        best_masks = [m for m in masks]
        
        # Greedy Local Search
        # We will iterate through all possible edge flips and pick the one that reduces violations the most.
        # If no improvement, we might get stuck. We can use a "kick" (random flips) to escape.
        
        no_improve_steps = 0
        max_no_improve = 50 # Adjust based on n
        
        while time.time() < end_time:
            if no_improve_steps > max_no_improve:
                # Perform a kick: flip 5 random edges
                for _ in range(5):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v: continue
                    if adj[u, v] == 1:
                        # Remove
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        adj[u, v] = 0
                        adj[v, u] = 0
                    else:
                        # Add
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        adj[u, v] = 1
                        adj[v, u] = 1
                current_violations = total_violations(n, masks)
                no_improve_steps = 0
                continue

            best_delta = 0
            best_u = -1
            best_v = -1
            best_adding = False
            
            # Evaluate all edge flips
            for u in range(n):
                for v in range(u + 1, n):
                    is_edge = adj[u, v] == 1
                    adding = not is_edge
                    
                    dk4 = delta_k4(n, masks, u, v, adding)
                    din6 = delta_ind6(n, masks, u, v, adding)
                    
                    if adding:
                        delta = dk4 + din6
                    else:
                        delta = -dk4 - din6
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_u = u
                        best_v = v
                        best_adding = adding
            
            if best_u == -1 or best_delta == 0:
                no_improve_steps += 1
                continue
            
            # Apply best flip
            u, v = best_u, best_v
            if best_adding:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                adj[u, v] = 1
                adj[v, u] = 1
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                adj[u, v] = 0
                adj[v, u] = 0
            
            current_violations += best_delta
            no_improve_steps = 0
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_masks = [m for m in masks]
                
                if current_violations == 0:
                    return get_edges_from_masks(n, masks)
        
        # Return best found
        if best_violations == 0:
            return get_edges_from_masks(n, best_masks)
        return None

    # Strategy: try to extend from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Allocate time
        if n <= 24:
            budget = min(remaining * 0.5, 10.0)
        elif n <= 30:
            budget = min(remaining * 0.4, 8.0)
        else:
            budget = min(remaining * 0.3, 5.0)
            
        found = None
        num_restarts = 2
        per_restart = budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() > deadline - 1.0:
                break
                
            if restart == 0 and n <= 25:
                # Try extending Paley(17)
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.5:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            else:
                found = try_find_graph(n, per_restart)
                
            if found is not None:
                break
        
        if found is not None:
            best_n = n
            best_edges = found
            if n >= 35:
                break
    
    return {'n': best_n, 'edges': best_edges}
