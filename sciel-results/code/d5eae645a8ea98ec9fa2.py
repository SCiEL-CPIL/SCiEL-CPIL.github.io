import random
import numpy as np

def construct():
    n = 20
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def check_k33_after_add(u, v, adj):
        # Check if adding edge (u, v) creates a K3,3
        # u and v must be in opposite parts of the K3,3
        # Part A contains u, Part B contains v
        # We need 2 more in A (from N(v)) and 2 more in B (from N(u))
        
        # Get neighbors
        # adj is 2D list or numpy array? Let's use list of lists for speed in Python loops if small, or numpy.
        # Numpy indexing is slow for single elements in Python loops.
        # Let's use a list of bitmasks (Python ints) for adjacency.
        # But the function signature takes adj as input in the seed. 
        # I will change the internal representation to bitmasks for speed.
        pass

    # Revised Plan: Use bitmasks internally.
    
    def check_k33_add(u, v, adj_masks):
        # adj_masks[i] is an integer bitmask of neighbors of i
        # We are adding edge u-v.
        # u is in A, v is in B.
        # Candidates for A (other than u): neighbors of v.
        # Candidates for B (other than v): neighbors of u.
        
        cand_a_mask = adj_masks[v] & ~(1 << u)
        cand_b_mask = adj_masks[u] & ~(1 << v)
        
        # Convert to lists of indices
        cand_a = [i for i in range(n) if (cand_a_mask >> i) & 1]
        cand_b = [i for i in range(n) if (cand_b_mask >> i) & 1]
        
        if len(cand_a) < 2 or len(cand_b) < 2:
            return True # Safe
        
        # Check all pairs
        for i in range(len(cand_a)):
            a1 = cand_a[i]
            for j in range(i+1, len(cand_a)):
                a2 = cand_a[j]
                # Check if a1, a2 can be in A
                # They must be connected to b1, b2 in B
                # Also a1, a2 must not be in B (i.e., not in cand_b? No, they just must be distinct from b1, b2)
                # Actually, if a1 is in cand_b, it means a1 is connected to u.
                # If a1 is chosen for A, and also b1 is chosen for B, and a1==b1, that's invalid.
                # So we need to ensure the sets are disjoint.
                
                # For each pair (a1, a2), find if there exist b1, b2 in cand_b
                # such that b1 != a1, b2 != a1, b1 != a2, b2 != a2
                # and a1, a2 connected to b1, b2.
                
                # Pre-filter cand_b: remove a1, a2
                valid_b_mask = cand_b_mask & ~(1 << a1) & ~(1 << a2)
                valid_b = [i for i in range(n) if (valid_b_mask >> i) & 1]
                
                if len(valid_b) < 2:
                    continue
                
                # Check connectivity
                # a1 must be connected to b1, b2
                # a2 must be connected to b1, b2
                
                # Get mask of neighbors of a1
                mask_a1 = adj_masks[a1]
                mask_a2 = adj_masks[a2]
                
                # We need b1, b2 from valid_b such that:
                # (mask_a1 has b1, mask_a1 has b2)
                # (mask_a2 has b1, mask_a2 has b2)
                
                # Iterate valid_b pairs
                for k in range(len(valid_b)):
                    b1 = valid_b[k]
                    if not ((mask_a1 >> b1) & 1):
                        continue
                    if not ((mask_a2 >> b1) & 1):
                        continue
                    for l in range(k+1, len(valid_b)):
                        b2 = valid_b[l]
                        if not ((mask_a1 >> b2) & 1):
                            continue
                        if not ((mask_a2 >> b2) & 1):
                            continue
                        # Found K3,3
                        return False
        return True

    best_count = 0
    best_edges = []
    
    # Run multiple random restarts
    for _ in range(100):
        adj_masks = [0] * n
        count = 0
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        for u, v in edges_copy:
            if check_k33_add(u, v, adj_masks):
                # Add edge
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                count += 1
                
        if count > best_count:
            best_count = count
            # Reconstruct edges list from masks
            temp_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_masks[i] >> j) & 1:
                        temp_edges.append([i, j])
            best_edges = temp_edges
            
        # Optional: Early stop if we hit a theoretical upper bound?
        # ex(20, K3,3) is likely < 60. If we hit 60, maybe stop?
        # But let's run all 100 to be safe, it's fast enough.
        
    return best_edges
