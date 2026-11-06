        # Count 4-cycles
        # A 4-cycle is u-v-w-x-u.
        # In a 3-regular graph, we can count them by checking pairs of neighbors.
        # For each edge (u,v), count common neighbors? No.
        # Number of 4-cycles = 1/8 * sum_{u,v} (common_neighbors(u,v) * ...)?
        # Simpler:
        # Iterate all pairs of vertices (u,v).
        # If u and v are not adjacent, count common neighbors.
        # Each 4-cycle u-v-w-x has two non-adjacent pairs (u,w) and (v,x).
        # So sum over all non-adjacent pairs of common neighbors / 2 = number of 4-cycles?
        # Let's verify.
        # Cycle 1-2-3-4-1.
        # Non-adj pairs: (1,3), (2,4).
        # Common neighbors of 1,3: {2,4}. Count 2.
        # Common neighbors of 2,4: {1,3}. Count 2.
        # Sum = 4.
        # Number of 4-cycles = 1.
        # So Sum / 4 = 1.
        
        common_neigh_sum = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (i, j) not in local_edges and (j, i) not in local_edges:
                    # Count common neighbors
                    cn = 0
                    for k in range(8):
                        if k != i and k != j:
                            if ((i, k) in local_edges or (k, i) in local_edges) and ((j, k) in local_edges or (k, j) in local_edges):
                                cn += 1
                    common_neigh_sum += cn
        num_4_cycles = common_neigh_sum // 4
        
        if num_4_cycles != 6:
            return False
