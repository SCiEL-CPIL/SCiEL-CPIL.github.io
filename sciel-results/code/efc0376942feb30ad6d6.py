import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 6-subsets of vertices
    # We need subsets T of size 6 from V \ {u, v}
    # It is more efficient to generate them on the fly or precompute all 6-subsets of V
    # and filter those not containing u, v?
    # Precomputing all C(16,6) = 8008 subsets is fine.
    all_6_subsets = list(itertools.combinations(vertices, 6))
    
    # Precompute partitions of a 6-element set into two 3-element sets
    # For a specific tuple T of 6 elements, we need all ways to split into A' and B' of size 3.
    # There are C(6,3)/2 = 10 unique splits (since A' and B' are symmetric in the check logic mostly, 
    # but here u and v are distinct, so we treat (A', B') and (B', A') as distinct cases 
    # corresponding to u in A or v in A).
    # Actually, for a fixed T, we iterate over all C(6,3) = 20 subsets A' of T.
    # B' is the complement.
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) to graph adj creates a Q3.
        adj is a list of 16 integers, where adj[i] is a bitmask of neighbors of i.
        """
        # We only need to check 8-vertex subsets S that contain u and v.
        # S = T U {u, v}, where T is a 6-subset of V \ {u, v}.
        
        # To speed up, we can iterate over all 6-subsets T of V.
        # If u in T or v in T, skip.
        
        for T in all_6_subsets:
            if u in T or v in T:
                continue
            
            # T is a tuple of 6 vertices.
            # We need to check if there is a bipartition of S = T U {u, v} 
            # such that S induces Q3 and (u, v) is an edge in Q3.
            # This implies u and v are in different parts.
            
            # Iterate over all subsets A' of T of size 3.
            # There are 20 such subsets.
            # For each A', let B' be the complement in T.
            
            # Case 1: u in A, v in B.
            # A = A' U {u}, B = B' U {v}.
            # Conditions for Q3:
            # 1. deg(u, B) = 3. Since (u, v) is added, deg(u, B') + 1 = 3 => deg(u, B') = 2.
            # 2. deg(v, A) = 3. Since (u, v) is added, deg(v, A') + 1 = 3 => deg(v, A') = 2.
            # 3. For x in A', deg(x, B) = 3. B = B' U {v}.
            #    deg(x, B') + (1 if x~v else 0) = 3.
            # 4. For y in B', deg(y, A) = 3. A = A' U {u}.
            #    deg(y, A') + (1 if y~u else 0) = 3.
            
            # We can precompute masks for T?
            # Let's just loop.
            
            T_list = list(T)
            # Generate all 3-subsets of T_list
            for A_prime in itertools.combinations(T_list, 3):
                B_prime = tuple(x for x in T_list if x not in A_prime)
                
                # Check Case 1: u in A, v in B
                # deg(u, B') must be 2
                mask_B_prime = 0
                for b in B_prime:
                    mask_B_prime |= (1 << b)
                if (adj[u] & mask_B_prime).bit_count() != 2:
                    # Note: adj[u] does NOT contain v yet.
                    pass
                else:
                    # deg(v, A') must be 2
                    mask_A_prime = 0
                    for a in A_prime:
                        mask_A_prime |= (1 << a)
                    if (adj[v] & mask_A_prime).bit_count() != 2:
                        pass
                    else:
                        # Check degrees of vertices in A' and B'
                        valid = True
                        # For x in A', deg(x, B') + (x~v) == 3
                        for x in A_prime:
                            deg_x_B_prime = (adj[x] & mask_B_prime).bit_count()
                            x_v_edge = 1 if (adj[x] & (1 << v)) else 0
                            if deg_x_B_prime + x_v_edge != 3:
                                valid = False
                                break
                        if valid:
                            # For y in B', deg(y, A') + (y~u) == 3
                            for y in B_prime:
                                deg_y_A_prime = (adj[y] & mask_A_prime).bit_count()
                                y_u_edge = 1 if (adj[y] & (1 << u)) else 0
                                if deg_y_A_prime + y_u_edge != 3:
                                    valid = False
                                    break
                        if valid:
                            return False # Q3 found, cannot add
                
                # Check Case 2: v in A, u in B
                # A = A' U {v}, B = B' U {u}
                # deg(v, B') must be 2
                if (adj[v] & mask_B_prime).bit_count() != 2:
                    pass
                else:
                    # deg(u, A') must be 2
                    if (adj[u] & mask_A_prime).bit_count() != 2:
                        pass
                    else:
                        valid = True
                        # For x in A', deg(x, B) = deg(x, B' U {u}) = deg(x, B') + (x~u)
                        for x in A_prime:
                            deg_x_B_prime = (adj[x] & mask_B_prime).bit_count()
                            x_u_edge = 1 if (adj[x] & (1 << u)) else 0
                            if deg_x_B_prime + x_u_edge != 3:
                                valid = False
                                break
                        if valid:
                            # For y in B', deg(y, A) = deg(y, A' U {v}) = deg(y, A') + (y~v)
                            for y in B_prime:
                                deg_y_A_prime = (adj[y] & mask_A_prime).bit_count()
                                y_v_edge = 1 if (adj[y] & (1 << v)) else 0
                                if deg_y_A_prime + y_v_edge != 3:
                                    valid = False
                                    break
                        if valid:
                            return False # Q3 found, cannot add
        
        return True

    # Initial Graph: Random dense
    adj = [0] * n
    edges = set()
    
    # Start with a random graph with p=0.7
    random.seed(42)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.7:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
    
    best_edges = set(edges)
    best_count = len(edges)
    
    # Local Search
    # We try to add edges.
    # We do multiple passes.
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        # Optional: If stuck, do a kick?
        # For now, just greedy addition.
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Shuffle non-edges to avoid bias
            non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edges]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline - 5:
                    break
                if can_add_edge(adj, u, v):
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
                    improved = True
                    
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            
        # If we found a good solution, we might try to restart from scratch with different seed
        # or just keep improving.
        # Let's try a kick if we are stuck.
        if not improved and len(edges) > 0:
             # Remove 3 random edges
             if len(edges) > 10:
                 to_remove = random.sample(list(edges), 3)
                 for u, v in to_remove:
                     adj[u] &= ~(1 << v)
                     adj[v] &= ~(1 << u)
                     edges.remove((u, v))

    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
