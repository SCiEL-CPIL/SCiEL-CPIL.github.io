import time
import random
import sys
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Random search for n=20, verified by exhaustive subset check using optimized bit operations.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_n = 20
    best_n = 5
    best_edges = []
    
    # Helper to check if a specific graph (list of edge masks) is valid
    def is_graph_valid(adj_masks, n):
        """
        adj_masks: list of n ints, bitmask of neighbors for each vertex.
        Returns True if no subset of size >= 6 induces a regular subgraph.
        """
        # Precompute popcount for all masks? No, use int.bit_count()
        
        # Iterate over all subsets of {0..n-1}
        # We only care about subsets of size >= 6
        # Total subsets 2^n. For n=20, 1,048,576.
        
        # Optimization: 
        # We can iterate by size? No, just iterate all and check size.
        # To speed up, we can skip sizes < 6.
        
        # A subset S is regular if:
        # 1. All vertices in S have the same degree within S.
        # 2. That degree is d.
        
        # For a given S (bitmask):
        # d_u = popcount(adj_masks[u] & S) for u in S.
        # All d_u must be equal.
        
        # We can break early if we find a regular subset.
        
        # To make this faster in Python, we can use a trick:
        # If S is regular, its complement? No.
        
        # Let's just iterate.
        # 1M iterations * 20 ops = 20M ops. ~ 1s in Python.
        # This allows ~100 checks in 100s.
        
        for S in range(1, 1 << n):
            # Check size
            size = S.bit_count()
            if size < 6:
                continue
            
            # Check regularity
            # Get the first vertex in S
            # u0 = (S & -S).bit_length() - 1
            lsb = S & -S
            u0 = (lsb.bit_length() - 1)
            
            d_target = (adj_masks[u0] & S).bit_count()
            
            # Check all other vertices in S
            # We can iterate through bits of S
            temp_S = S
            valid = True
            while temp_S:
                lsb = temp_S & -temp_S
                u = lsb.bit_length() - 1
                if u == u0:
                    temp_S ^= lsb
                    continue
                d_u = (adj_masks[u] & S).bit_count()
                if d_u != d_target:
                    valid = False
                    break
                temp_S ^= lsb
            
            if valid:
                return False
        
        return True

    # Main Loop: Random Search
    # Try to find a graph on 20 vertices.
    # If not found, try 19, 18, etc.
    
    for n_target in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        # Limit attempts to avoid infinite loop if probability is low
        max_attempts = 5000 
        # But we must respect time.
        
        while not found and time.time() < deadline:
            attempts += 1
            
            # Generate random graph
            # Probability 0.5 is standard
            adj_masks = []
            for i in range(n_target):
                mask = 0
                # Randomly connect to previous vertices
                for j in range(i):
                    if random.random() < 0.5:
                        mask |= (1 << j)
                adj_masks.append(mask)
            
            # Check validity
            if is_graph_valid(adj_masks, n_target):
                # Convert to edge list
                edges = []
                for i in range(n_target):
                    mask = adj_masks[i]
                    for j in range(i):
                        if (mask >> j) & 1:
                            edges.append([j, i])
                
                if n_target > best_n:
                    best_n = n_target
                    best_edges = edges
                
                found = True
                break
        
        if found:
            break # Found a graph for this n, move to return

    # Fallback: If we didn't find n=20, return the best we found (likely n=5 from init or smaller valid ones)
    # To ensure we return *something* valid, we should verify the best_edges if we updated it.
    # The loop above only updates best_edges if is_graph_valid returned True.
    # So best_edges is always valid for best_n.
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
