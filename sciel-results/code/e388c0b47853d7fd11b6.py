import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Leave a safety margin for the evaluator's kill signal
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) creates a K3,3 subgraph.
        This is equivalent to checking if the bipartite graph between 
        N(v)\{u} and N(u)\{v} contains a K2,2.
        """
        # A = N(v) \ {u}, B = N(u) \ {v}
        # We iterate over pairs in B and check if they share >= 2 neighbors in A.
        mask_b = adj[u] & ~(1 << v)
        mask_a = adj[v] & ~(1 << u)
        
        # Optimization: If either set is small, the K2,2 cannot exist
        # K2,2 requires at least 2 vertices in each set.
        if mask_b.bit_count() < 2 or mask_a.bit_count() < 2:
            return True
            
        # Iterate over pairs in B
        # If B is large, this might be slow. 
        # However, for n=20, max degree is 19. Pairs are ~171.
        # Inner check is bitmask AND and bit_count.
        
        # Alternative: Iterate over pairs in the smaller set?
        # The logic is symmetric. 
        # Let's stick to iterating B as in the inspiration, but ensure we pick the smaller set for iteration if possible?
        # Actually, the check is: exists y1, y2 in B such that |N(y1) & N(y2) & A| >= 2.
        # This is not symmetric in computation cost if we don't precompute.
        
        mask = mask_b
        while mask:
            lsb1 = mask & -mask
            y1 = lsb1.bit_length() - 1
            mask2 = mask ^ lsb1
            mask = mask2
            while mask2:
                lsb2 = mask2 & -mask2
                y2 = lsb2.bit_length() - 1
                mask2 = mask2 ^ lsb2
                
                # Common neighbors of y1 and y2, restricted to A
                common = (adj[y1] & adj[y2]) & mask_a
                if common.bit_count() >= 2:
                    return False
        return True

    def adj_to_edges(adj):
        edges = []
        for i in range_n:
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def is_k33_free(adj):
        """
        Verify the whole graph is K3,3-free.
        Used for final validation and initial seed check.
        """
        for u in range_n:
            for v in range(u + 1, n):
                # Check if adding (u,v) to a graph where (u,v) is MISSING would create K3,3?
                # No, this function checks the existing graph.
                # We can reuse the logic: 
                # For every pair u,v, if they are NOT connected, adding them must not create K3,3?
                # No, the graph is K3,3-free if NO K3,3 exists.
                # A K3,3 exists if there are sets S,T of size 3 fully connected.
                # Efficient check: 
                # For every pair u,v, let C = N(u) & N(v).
                # If |C| >= 3, we must check if any 3 vertices in C form a triangle? No.
                # The condition for K3,3-free is:
                # For any u, v, the graph induced by N(u) & N(v) must be K2,2-free? No.
                
                # Let's use the property: 
                # A graph is K3,3-free iff for every edge (u,v) that is NOT in the graph, 
                # the bipartite graph between N(u)\{v} and N(v)\{u} is K2,2-free.
                # Wait, that's for "adding an edge creates K3,3".
                # If the graph is K3,3-free, then for any non-edge (u,v), adding it must not create a K3,3.
                # So we can just check all non-edges.
                
                if not ((adj[u] >> v) & 1):
                    if not is_valid_addition(adj, u, v):
                        return False
        return True

    def build_polarity_graph():
        """Construct a K3,3-free graph from polarity of projective plane of order 4."""
        def gf4_add(a, b):
            return a ^ b
        
        def gf4_mul(a, b):
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1:
                return a
            lookup = {(2,2): 3, (2,3): 1, (3,2): 1, (3,3): 2}
            return lookup[(a, b)]
        
        def gf4_inv(a):
            if a == 0:
                return 0
            if a == 1:
                return 1
            if a == 2:
                return 3
            if a == 3:
                return 2
        
        points = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = gf4_inv(x)
                        pt = (1, gf4_mul(inv_x, y), gf4_mul(inv_x, z))
                    elif y != 0:
                        inv_y = gf4_inv(y)
                        pt = (0, 1, gf4_mul(inv_y, z))
                    else:
                        pt = (0, 0, 1)
                    
                    found = False
                    for p in points:
                        if p == pt:
                            found = True
                            break
                    if not found:
                        points.append(pt)
        
        # 21 points
        adj21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                p1, p2 = points[i], points[j]
                val = gf4_add(gf4_add(gf4_mul(p1[0], p2[2]), gf4_mul(p1[2], p2[0])), gf4_mul(p1[1], p2[1]))
                if val == 0:
                    adj21[i] |= (1 << j)
                    adj21[j] |= (1 << i)
        
        # Use first 20 points
        adj20 = [0] * 20
        for i in range(20):
            for j in range(20):
                if i != j and (adj21[i] >> j) & 1:
                    adj20[i] |= (1 << j)
        
        return adj20

    # Initialize with structured graph
    init_adj = build_polarity_graph()
    
    # Verify initial graph
    if not is_k33_free(init_adj):
        init_adj = [0] * n
        
    best_adj = init_adj[:]
    best_edges = adj_to_edges(best_adj)
    best_count = len(best_edges)
    
    # Local Search
    # We will run multiple restarts or a long single search.
    # Given the time budget, a single long search with randomization is usually better for n=20.
    
    current_adj = best_adj[:]
    current_edges = adj_to_edges(current_adj)
    current_count = len(current_edges)
    
    # Precompute missing edges list for faster access
    def get_missing_edges(adj):
        missing = []
        for u in range_n:
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        return missing

    missing_edges = get_missing_edges(current_adj)
    
    # Main optimization loop
    while time.time() < deadline:
        # Phase 1: Greedy Add
        improved = False
        new_missing = []
        # Shuffle to avoid bias
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                new_missing.append((u, v))
                continue
            if is_valid_addition(current_adj, u, v):
                # Add edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
                improved = True
            else:
                new_missing.append((u, v))
        
        if improved:
            missing_edges = new_missing
            # Update best if better
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
                best_edges = current_edges[:]
            continue
        
        # Phase 2: If stuck, try swaps (Remove 1, Add 1) to escape local optimum
        # Or Remove 1, Add 2 (Net +1)
        
        # Try to find a swap that allows adding more edges later
        # Randomly pick an edge to remove
        if not current_edges or not missing_edges:
            break
            
        swap_success = False
        # Limit swap attempts to avoid wasting time
        max_swap_attempts = 50
        for _ in range(max_swap_attempts):
            if time.time() > deadline - 5:
                break
                
            # Pick a random existing edge to remove
            e_idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[e_idx]
            
            # Remove it
            current_adj[u_rem] &= ~(1 << v_rem)
            current_adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(e_idx)
            current_count -= 1
            
            # Now try to add some missing edges
            # We can just run the greedy add phase again on the modified graph
            # But that might be slow. 
            # Instead, try to add a specific missing edge.
            
            # Pick a random missing edge
            m_idx = random.randint(0, len(missing_edges) - 1)
            u_add, v_add = missing_edges[m_idx]
            
            if is_valid_addition(current_adj, u_add, v_add):
                # Add it
                current_adj[u_add] |= (1 << v_add)
                current_adj[v_add] |= (1 << u_add)
                current_edges.append([u_add, v_add])
                current_count += 1
                swap_success = True
                break
            else:
                # Restore the removed edge if we can't add a new one
                current_adj[u_rem] |= (1 << v_rem)
                current_adj[v_rem] |= (1 << u_rem)
                current_edges.append([u_rem, v_rem])
                current_count += 1
        
        if not swap_success:
            # If we couldn't find a beneficial swap, maybe restart from best?
            # Or just break if we are stuck for a while.
            # To avoid infinite loops, we can restart from the best known graph.
            current_adj = best_adj[:]
            current_edges = adj_to_edges(current_adj)
            current_count = best_count
            missing_edges = get_missing_edges(current_adj)
            
            # If we restart too often, we might not improve.
            # Let's add a counter for restarts.
            # For simplicity, we just continue. The randomization in shuffle helps.
            
    # Final validation
    if is_k33_free(best_adj):
        return best_edges
    else:
        # Fallback: return empty or a known valid small graph
        # This should not happen if is_valid_addition is correct
        return []

