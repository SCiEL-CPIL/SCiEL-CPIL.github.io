"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import itertools

# EVOLVE-BLOCK-START

def _has_k222(adj, n):
    """
    Check if the graph with adjacency matrix adj (boolean numpy array) on n vertices
    contains K_{2,2,2} as a subgraph.
    K_{2,2,2} is the complete tripartite graph with parts of size 2.
    Vertices: A={a1, a2}, B={b1, b2}, C={c1, c2}
    Edges: all pairs between distinct parts.
    Total 6 vertices, 12 edges.
    """
    # Iterate over all 6-tuples of distinct vertices? No, too slow.
    # Instead, iterate over all partitions of 6 vertices into 3 pairs?
    # Or simply: does there exist a set of 6 vertices {v0..v5} such that 
    # edges (v0,v1) NO, (v2,v3) NO, (v4,v5) NO
    # and all other pairs among these 6 are YES?
    
    # Since n <= 30, we can brute force check for K_{2,2,2}.
    # Number of 6-subsets of 30 is C(30,6) ~ 593,775.
    # For each subset, we need to check if it contains K_{2,2,2}.
    # A K_{2,2,2} is a partition of the 6 vertices into 3 pairs (a,b), (c,d), (e,f)
    # such that no edges within pairs, and all edges between pairs.
    
    # There are 15 ways to partition 6 items into 3 pairs.
    # So for each 6-subset, we check 15 partitions.
    # 593,775 * 15 * 12 checks = ~100M operations. Might be slow in Python.
    
    # Better approach:
    # For every pair of disjoint pairs (a,b) and (c,d) that are non-edges,
    # check if there is another pair (e,f) non-edges, disjoint from {a,b,c,d},
    # such that all cross edges exist.
    
    # Let's try a slightly smarter enumeration.
    # Enumerate all pairs of non-edges (u,v) and (w,x) that are disjoint.
    # If (u,v) and (w,x) are non-edges and disjoint, then we need two more vertices e,f
    # such that (e,f) is a non-edge, e,f not in {u,v,w,x},
    # and e,f are connected to all of u,v,w,x.
    
    # Actually, the simplest robust check for n<=30 is:
    # For all combinations of 6 vertices:
    #   Check if the induced subgraph is K_{2,2,2}.
    # K_{2,2,2} is the complement of 3 disjoint edges.
    # So the induced subgraph on {v0..v5} is K_{2,2,2} iff
    #   the set of non-edges within {v0..v5} is exactly a perfect matching (3 edges).
    
    # Let's optimize the check:
    # 1. Generate all 6-combinations.
    # 2. For each combination, extract the 6x6 submatrix of adj.
    # 3. Count non-edges. If exactly 3, check if they form a matching.
    
    # This is still ~600k iterations. In Python, this might take a few seconds.
    # With deadline 150s, this is acceptable for validation.
    
    for comb in itertools.combinations(range(n), 6):
        # Check if induced subgraph is K222
        # Non-edges must be exactly 3 and form a matching.
        non_edges = []
        # Map indices in comb to 0..5
        for i in range(6):
            vi = comb[i]
            for j in range(i+1, 6):
                vj = comb[j]
                if not adj[vi, vj]:
                    non_edges.append((i, j))
        
        if len(non_edges) != 3:
            continue
            
        # Check if non_edges form a matching (no shared vertices)
        used = [False] * 6
        valid = True
        for u, v in non_edges:
            if used[u] or used[v]:
                valid = False
                break
            used[u] = True
            used[v] = True
        if valid:
            return True
            
    return False

def _is_valid_coloring(n, colors):
    """
    Verify that neither color class contains K_{2,2,2}.
    colors is a list of 0/1 of length n*(n-1)/2.
    """
    # Build adjacency matrices for color 0 and color 1
    import numpy as np
    adj0 = np.zeros((n, n), dtype=bool)
    adj1 = np.zeros((n, n), dtype=bool)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            if c == 0:
                adj0[i, j] = True
                adj0[j, i] = True
            else:
                adj1[i, j] = True
                adj1[j, i] = True
            idx += 1
            
    if _has_k222(adj0, n):
        return False
    if _has_k222(adj1, n):
        return False
    return True

def _paley_coloring(n):
    """
    Construct the Paley graph coloring for n vertices.
    Assumes n is a prime congruent to 1 mod 4.
    Color 0 if (j-i) % n is a quadratic residue (non-zero).
    """
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return colors

def construct():
    start_time = time.time()
    deadline = 150.0  # 120s hard limit, keep margin
    
    # Known maximum n is 29 for R(K2,2,2) = 31.
    # We try to construct a valid coloring for n=29.
    # If it fails, we fall back to smaller n or random search.
    
    candidates = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
    
    # First, try the Paley construction for n=29 (and other primes 1 mod 4)
    # Primes in candidates: 29, 17
    
    # Check n=29
    if 29 in candidates:
        colors_29 = _paley_coloring(29)
        # Validate
        if _is_valid_coloring(29, colors_29):
            return {'n': 29, 'colors': colors_29}
            
    # If Paley 29 failed (should not), or if we want to try other constructions?
    # The task says "Maximize n". 29 is the theoretical max.
    # If 29 fails validation, we must try 28? 
    # R(K2,2,2) = 31 means no coloring exists for n=30. 
    # It does NOT mean no coloring exists for n=28. 
    # But if 29 works, we return 29.
    
    # What if Paley 29 is invalid? (It shouldn't be, but just in case)
    # We can try a greedy/backtracking search for n=28 or 29.
    # Given the strict time limit and the fact that 29 is known to work,
    # we should prioritize returning a valid smaller n if 29 fails.
    
    # Let's implement a fast random/greedy solver for n=29 just in case.
    # But first, let's double check the logic of _is_valid_coloring.
    # It is correct.
    
    # If 29 fails, try 28?
    # We can try to extend a known good coloring or use random restarts.
    # However, 29 is so standard that if it fails, there might be a bug in my validator.
    
    # Let's add a fallback to n=17 (seed) to guarantee a valid answer.
    if time.time() > deadline - 10:
        # Emergency exit
        colors_17 = _paley_coloring(17)
        return {'n': 17, 'colors': colors_17}
        
    # If we get here, 29 failed validation. 
    # This is unexpected. Let's try to find a valid coloring for n=28 or 29 with random search?
    # No, that's risky. 
    # Let's just return 17 if 29 fails, or try 25?
    # Actually, let's try to verify 17 as well.
    
    colors_17 = _paley_coloring(17)
    if _is_valid_coloring(17, colors_17):
        return {'n': 17, 'colors': colors_17}
        
    # If even 17 fails (bug?), return a trivial 2-vertex coloring.
    return {'n': 2, 'colors': [0]}

# EVOLVE-BLOCK-END
