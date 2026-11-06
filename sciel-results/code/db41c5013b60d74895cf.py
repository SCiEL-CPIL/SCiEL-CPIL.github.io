# EVOLVE-BLOCK-START
import random
import time
import itertools
import math
import numpy as np

def _is_regular_subset_fast(adj, n, subset, mask):
    """Check if the induced subgraph on 'subset' is regular."""
    if len(subset) < 6:
        return False
    d0 = -1
    for v in subset:
        d = (adj[v] & mask).bit_count()
        if d0 == -1:
            d0 = d
        elif d != d0:
            return False
    return True

def _has_violation(adj, n):
    """
    Check if the graph has any regular induced subgraph of size >= 6.
    Returns True if a violation is found, False otherwise.
    """
    # We only need to check subsets of size exactly 6? 
    # If a subset of size K > 6 is regular, it contains many subsets of size 6.
    # However, a subset of size 6 being regular is a violation.
    # If we find NO subset of size 6 is regular, does that mean no subset of size > 6 is regular?
    # No. A regular graph of size 7 might not contain a regular induced subgraph of size 6.
    # Example: C7 (cycle of 7). Degrees are all 2. Is there a regular induced subgraph of size 6?
    # Removing one vertex from C7 gives a path P6. Degrees in P6: 1,2,2,2,2,1. Not regular.
    # So checking only size 6 is NOT sufficient. We must check all sizes >= 6.
    # However, for random graphs, violations are usually at size 6.
    # To be safe and correct, we check all sizes.
    # Optimization: If we find a violation at size 6, we can stop.
    # If we don't find one at size 6, we must check 7, etc.
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset_fast(adj, n, subset, mask):
                return True
    return False

def _generate_random_adj(n, p=0.5):
    """Generate a random graph adjacency list."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Fallback: n=5 is always valid (no subsets of size >= 6)
    # We can construct a simple graph on 5 vertices, e.g., empty graph.
    best_edges = []
    
    # Try to find larger n
    # We prioritize n=21, then 20, etc.
    # But n=21 is hard. n=20 is known to exist.
    # Let's try random search for n=20 first, as it's more likely to succeed quickly.
    # Then try n=21.
    
    target_ns = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    for n_target in target_ns:
        if time.time() > deadline - 10:
            break
            
        # Try a few random graphs for this n
        # Number of attempts depends on remaining time
        attempts = 0
        while time.time() < deadline - 15 and attempts < 500:
            attempts += 1
            # Use varying probabilities
            p = random.uniform(0.3, 0.7)
            adj = _generate_random_adj(n_target, p)
            
            # Verify
            if not _has_violation(adj, n_target):
                # Found a valid graph
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                
                if n_target > best_n:
                    best_n = n_target
                    best_edges = edges
                
                # If we found a valid graph for this n, we can break to try larger n?
                # No, we are iterating from high to low.
                # If we found n=20, we break the inner loop and go to next n_target (19)?
                # No, we want the MAX n. So if we found 20, we should stop trying 19.
                # But we started with 21. If 21 failed, we tried 20. If 20 succeeds, we break.
                break
        
        # If we found a valid graph for the current n_target, we are done (since we go high to low)
        if best_n == n_target:
            break
            
    # If best_n is still 5, return the empty graph on 5 vertices
    # But we should ensure best_edges matches best_n
    if best_n == 5 and not best_edges:
        # Empty graph on 5 vertices
        best_edges = []
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
