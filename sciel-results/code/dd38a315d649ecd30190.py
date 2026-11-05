import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_matrix(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _check_validity_numpy(adj, n):
    """
    Check if graph has no regular induced subgraph on 6+ vertices.
    Returns True if valid (no violations), False otherwise.
    Uses vectorized NumPy for speed.
    """
    # Precompute all subsets of size 6 to 22? That's too many.
    # But we only need to check if ANY subset is regular.
    # A regular induced subgraph on k vertices means all degrees within that set are equal.
    # For k=6, the degree can be 0, 1, 2, 3, 4, or 5.
    # If a graph has no regular induced subgraph of size 6, it might still have one of size 7+.
    # However, if a subset of size > 6 is regular, any subset of size 6 within it is NOT necessarily regular.
    # Wait, if a graph on k vertices is regular, it doesn't imply its subgraphs are regular.
    # So we must check all sizes >= 6.
    
    # Strategy:
    # 1. Check all subsets of size 6. If none are regular, is it possible a larger one is?
    # Yes. E.g., a 7-vertex graph could be 3-regular, but all its 6-vertex subsets are not 3-regular (e.g., degrees 2,2,3,3,3,3).
    # So we must check all sizes.
    
    # Optimization:
    # If a subset S is regular, then for every v in S, deg_S(v) is constant.
    
    # Let's iterate by size.
    # For n=20, number of subsets of size 6 is C(20,6) = 38760.
    # Size 7: C(20,7) = 77520.
    # ...
    # Total subsets with size >= 6 is 2^20 - sum_{i=0}^5 C(20,i) ~ 1M.
    # Checking each one in Python is slow.
    
    # Vectorized approach for size 6:
    # Generate all 6-combinations.
    # For each combination, compute degrees.
    # This is still O(C(n,6) * 6 * 6) operations if not careful.
    
    # Better:
    # For a fixed set of vertices V, the degree of v in V is (A[V, :].sum(axis=1))[v].
    
    # Let's try a hybrid:
    # 1. Fast check for size 6 using precomputed masks or vectorization.
    # 2. If size 6 is clean, check larger sizes.
    
    # Actually, random graphs G(n, 0.5) are very likely to have NO regular induced subgraphs of size 6.
    # The probability that a specific 6-set is regular is small.
    # If we check size 6 and find none, we can be fairly confident, but must verify larger.
    
    # Let's just do a full check but optimized.
    
    # For each size k from 6 to n:
    #   For each subset S of size k:
    #     Compute degrees of vertices in S induced by S.
    #     Check if all equal.
    
    # To speed up, we can use the fact that if a graph is regular of size k,
    # the sum of degrees is k * d, which must be even.
    
    # Let's implement a check that iterates subsets but uses numpy for degree calculation.
    
    # However, iterating 1M subsets in Python is ~10-20s.
    # We have 120s budget.
    
    # Let's try to optimize the inner loop.
    
    # Alternative:
    # Use the property that if a subset is regular, it is a "regular induced subgraph".
    
    # Let's just write a reasonably fast checker.
    
    # Precompute adjacency as int masks for fast degree counting?
    # adj_masks[i] = bitmask of neighbors.
    # deg(v, S) = (adj_masks[v] & S_mask).bit_count()
    
    adj_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_masks[i] = mask
        
    # Iterate all masks
    # We can prune: if a mask is regular, return False.
    
    # To make it faster, we can check smaller sizes first?
    # No, any size >= 6.
    
    # Let's stick to the bitmask method but optimized.
    # The previous _is_valid_graph_fast was decent but maybe we can make it faster.
    
    # One optimization:
    # If a subset S is regular, then for any v in S, deg(v) is the same.
    # We can pick the first vertex v0 in S. Let d = deg(v0).
    # Then all other vertices in S must have degree d.
    
    # We can iterate masks.
    
    # But 2^20 is 1,048,576.
    # For each mask, we iterate its bits.
    # Average bits ~ n/2 = 10.
    # So ~10M bit_count operations.
    # Python bit_count is fast (C level).
    # 10M operations should take < 1s? Maybe 2-3s.
    # So we can afford to check validity multiple times.
    
    # Let's reuse the logic from the seed but ensure it's tight.
    
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
            
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        if not verts:
            continue
            
        first_deg = (adj_masks[verts[0]] & mask).bit_count()
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
                
        if is_regular:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_edges = []
    
    # Try n=20, 21, 22
    for target_n in [20, 21, 22]:
        if time.time() > deadline - 5:
            break
            
        n = target_n
        found = False
        
        # Random restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            # Use p=0.5
            adj = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
            # Check validity
            if _check_validity_numpy(adj, n):
                best_n = n
                best_edges = _edges_from_adj_matrix(adj, n)
                found = True
                break
                
        if found:
            break
            
    if best_n == 0:
        # Fallback to a small valid graph
        # A path graph P_n has no regular induced subgraph of size >= 6?
        # P_6: degrees 1,2,2,2,2,1. Not regular.
        # Any induced subgraph of a path is a forest.
        # A regular forest? Only K_1 (degree 0) or K_2 (degree 1) or...
        # A cycle is regular but not a forest.
        # So any forest has no regular induced subgraph of size >= 3?
        # Wait, K_2 is regular (degree 1). Size 2.
        # K_1 is regular (degree 0). Size 1.
        # For size >= 3, a forest cannot be regular?
        # If a forest is regular of degree d:
        # d=0: isolated vertices. Size k. Regular.
        # So a set of k isolated vertices is a regular induced subgraph.
        # So a graph with k isolated vertices has a regular induced subgraph of size k.
        # To avoid this, we need no set of >= 6 isolated vertices.
        # A path graph has no isolated vertices.
        # Does a path graph have any regular induced subgraph of size >= 6?
        # Induced subgraphs of a path are disjoint unions of paths.
        # A disjoint union of paths is regular iff all components are isomorphic and the degree is constant.
        # In a path, degrees are 1 (ends) and 2 (internal).
        # So a regular induced subgraph must have all vertices degree 1 or all degree 2?
        # If all degree 1, it's a matching. Max matching size in a path is floor(n/2).
        # So if n >= 12, we can have a matching of size 6 (12 vertices).
        # So P_12 has a regular induced subgraph of size 12 (6 disjoint edges).
        # So we need to be careful.
        
        # Let's just return a small n if we fail.
        # n=5 is always safe?
        # We need no regular induced subgraph of size >= 6.
        # If n=5, there are no subsets of size >= 6. So it's valid.
        # The task says "Maximize n".
        # If we can't find 20, we should return the best we found.
        # But we initialized best_n=0.
        
        # Let's ensure we always return something valid.
        # If the loop finishes without finding a valid graph for 20, 21, 22,
        # we should try smaller n?
        # Or just return n=5?
        # The score is n.
        # If we return n=5, score is 5.
        # If we return n=20, score is 20.
        
        # The seed program returned n=5 if best_n <= 5.
        # But it tried to find 20.
        
        # Let's add a fallback loop for n=5..19 if we fail for 20.
        # But random graphs for small n might also fail?
        # No, for n < 6, it's always valid.
        # For n=6, we need a graph on 6 vertices that is NOT regular.
        # Most random graphs on 6 vertices are not regular.
        # So we can easily find valid graphs for small n.
        
        if best_n == 0:
            for target_n in range(19, 4, -1):
                if time.time() > deadline - 2:
                    break
                n = target_n
                for r in range(10):
                    if time.time() > deadline - 2:
                        break
                    adj = np.zeros((n, n), dtype=int)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                adj[i, j] = 1
                                adj[j, i] = 1
                    if _check_validity_numpy(adj, n):
                        best_n = n
                        best_edges = _edges_from_adj_matrix(adj, n)
                        break
                if best_n > 0:
                    break
            
            if best_n == 0:
                # Absolute fallback: n=5, no edges
                best_n = 5
                best_edges = []

    return {'n': best_n, 'edges': best_edges}
