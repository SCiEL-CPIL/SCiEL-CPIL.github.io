"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance 
    is at least the maximum vector norm.
    
    Strategy:
    Instead of using a greedy heuristic on a random set of vectors, we use a 
    structured construction derived from the Leech lattice properties.
    We look for vectors of the form with exactly 4 non-zero entries, each being 
    +/- 2. This corresponds to norm^2 = 4*4 = 16.
    If we can find a set of such vectors where the inner product is always 
    0, 4, or -4 (i.e., |ip| <= 4), then the distance squared between any two 
    vectors u, v is ||u||^2 + ||v||^2 - 2*u.v = 16 + 16 - 2*ip = 32 - 2*ip.
    The minimum distance squared is minimized when ip is maximized.
    If max ip = 4, min dist^2 = 32 - 8 = 24.
    The max norm squared is 16.
    The condition is min_dist >= max_norm, i.e., min_dist^2 >= max_norm^2.
    24 >= 16 holds.
    
    So we need a set of vectors in {0, +/-2}^11 with exactly four non-zeros, 
    such that for any pair, the inner product is in {-4, 0, 4}.
    This is equivalent to finding a large set of 4-subsets of {1..11} with 
    signs such that the overlap of supports is 0, 1, or 4?
    No, inner product is sum of products of non-zero entries.
    If supports overlap in k positions, the ip is sum of k terms of (+2)(+2)=4 or (+2)(-2)=-4.
    So ip is a multiple of 4.
    We require |ip| <= 4.
    This means the sum of the k terms must be +/-4 or 0.
    If k=0, ip=0. OK.
    If k=1, ip=+4 or -4. OK.
    If k=2, ip can be 8, 0, -8. We need 0. So one +4 and one -4.
    If k=3, ip can be 12, 4, -4, -12. We need +/-4. So two same, one opposite.
    If k=4, ip can be 16, 8, 0, -8, -16. We need 0? No, |ip|<=4. So ip must be 0? 
    Wait, if k=4, the supports are identical. The vectors are either identical or differ by signs.
    If identical, ip=16 (bad). If differ by 1 sign, ip=8 (bad). If differ by 2 signs, ip=0 (ok).
    If differ by 3 signs, ip=-8 (bad). If differ by 4 signs, ip=-16 (bad).
    
    So, we can have:
    - Disjoint supports (k=0)
    - Overlap 1 (any signs)
    - Overlap 2 (signs must be opposite in the two positions)
    - Overlap 3 (signs must be 2 same, 1 opposite)
    - Overlap 4 (signs must be 2 same, 2 opposite)
    
    This is a specific code construction. A simpler, robust approach for D=11:
    Use the set of all vectors with exactly 4 non-zero entries of value 1 or -1?
    Norm^2 = 4.
    Condition: min_dist^2 >= 4.
    dist^2 = 4 + 4 - 2*ip = 8 - 2*ip.
    We need 8 - 2*ip >= 4 => 2*ip <= 4 => ip <= 2.
    Also we need to check if ip can be negative? Yes.
    If ip = -2, dist^2 = 12 >= 4.
    So we need a set of vectors in {0, 1, -1}^11 with weight 4, such that 
    for any pair, inner product <= 2.
    
    The current program tries to find this via greedy. 
    However, we can use a known construction or a faster exact method for small D.
    Actually, for D=11, the kissing number is known to be at least 556 (or similar high number).
    The greedy method in the seed got ~1.26 * seed_score.
    
    Let's try a different structured approach:
    Generate all weight-4 vectors (5280 vectors).
    Use a fast, vectorized branch-and-bound or a simple "first fit" with sorting by degree?
    
    Actually, the prompt suggests a CSP/Backtracking approach. 
    Given N=5280, backtracking is too slow for full search.
    But we can use a "Conflict Graph" and find a large Independent Set using a 
    heuristic that is better than random greedy: 
    1. Compute the conflict graph (edge if ip > 2).
    2. Use a "Most Constrained Variable" heuristic for selection?
       In Independent Set, we want to pick vertices with fewest conflicts (low degree) first?
       Or most? 
       Usually, for Max Independent Set, picking low-degree vertices first allows more future picks.
       
    Let's implement a "Degree-Greedy" algorithm:
    1. Calculate degree of each vertex (number of conflicts).
    2. Pick the vertex with the lowest degree.
    3. Remove it and all its neighbors from the graph.
    4. Repeat.
    
    This is a standard heuristic (Greedily pick min-degree).
    We can optimize this with NumPy.
    
    Alternative: Use the structure of the graph.
    The graph is defined by supports.
    Vectors with disjoint supports are compatible.
    Vectors with 1 overlap are compatible.
    Vectors with 2 overlaps are compatible IF signs are opposite.
    Vectors with 3 overlaps are compatible IF 2 signs same, 1 opposite.
    Vectors with 4 overlaps (same support) are compatible IF 2 signs same, 2 opposite.
    
    We can group vectors by support (4-combination of indices).
    There are C(11,4) = 330 supports.
    For each support, there are 2^4 = 16 vectors.
    Two vectors with the same support are compatible if they differ in exactly 2 signs (ip=0).
    This forms a graph on the 16 vectors for each support.
    The max independent set for a single support (16 vectors, edge if ip>2 i.e. ip=4 or 16? 
    Wait, for same support, ip can be 16 (same), 8 (diff 1), 0 (diff 2), -8 (diff 3), -16 (diff 4).
    Conflict if ip > 2. So ip=16, 8 are conflicts.
    So we want ip <= 2.
    Compatible pairs in same support: ip=0 (diff 2 signs) or ip=-8 or -16?
    ip=-8 <= 2, so yes. ip=-16 <= 2, so yes.
    So any pair with diff >= 2 signs is compatible?
    Diff 2: ip=0. OK.
    Diff 3: ip=-8. OK.
    Diff 4: ip=-16. OK.
    So within a support, we can pick at most 8 vectors?
    The graph of conflicts for 16 vectors (hypercube Q4) where edge connects dist 0 and 1.
    We want independent set in this conflict graph, which is equivalent to finding a large set of vertices in Q4 with pairwise Hamming distance >= 2.
    This is a binary code of length 4, min distance 2.
    Max size is 2^(4-1) = 8.
    Example: All even parity vectors.
    
    So for each of the 330 supports, we can pick up to 8 vectors.
    Total potential: 330 * 8 = 2640.
    Now we need to handle conflicts between different supports.
    Two vectors with different supports S1, S2.
    Let k = |S1 intersect S2|.
    If k=0: ip=0. Always compatible.
    If k=1: ip=+4 or -4. Always compatible (<=2).
    If k=2: ip=8, 0, -8. Compatible if ip=0 or -8. i.e., signs must be opposite in the 2 shared positions.
    If k=3: ip=12, 4, -4, -12. Compatible if ip=4 or -4. i.e., 2 signs same, 1 opposite.
    
    This is a complex constraint satisfaction problem across the 330 supports.
    However, we can simplify by choosing a "base" set of vectors per support that minimizes cross-conflicts.
    For example, always choose the "all positive" vector for each support?
    If we choose all-positive for support S, it conflicts with any vector T that shares 2 or 3 positions with S and has same signs?
    If T shares 2 positions with S, and T is all-positive, ip=8 (conflict).
    So we can't just pick all-positive.
    
    Let's use the "Degree-Greedy" on the full graph of 5280 nodes.
    1. Build adjacency matrix or list. 5280 nodes.
    2. Iterate:
       a. Find node with min degree in remaining graph.
       b. Add to solution.
       c. Remove node and neighbors.
       
    This is O(N^2) or O(N * E).
    N=5280.
    We can implement this efficiently with NumPy.
    """
    
    deadline = time.time() + 110
    
    # Generate all weight-4 vectors
    vecs = []
    indices = range(D)
    for comb in combinations(indices, 4):
        for mask in range(16):
            v = np.zeros(D, dtype=np.int8)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    
    V = np.array(vecs, dtype=np.int8)
    N = len(V)
    
    # Compute inner products
    # We only care if ip > 2.
    # ip = V @ V.T
    # This is 5280x5280 int8 matrix. ~27 MB.
    ip_matrix = V @ V.T
    
    # Conflict matrix: True if ip > 2
    # Diagonal is 4, so True.
    conflict = ip_matrix > 2
    
    # Degree of each node: sum of conflicts (excluding self? self is conflict, but we remove self anyway)
    # Actually, we want to pick a node, then remove it and its neighbors.
    # So we need to know which nodes are "alive".
    
    alive = np.ones(N, dtype=bool)
    selected = []
    
    # To speed up, we can precompute the conflict neighbors for each node?
    # Or just use the matrix.
    # In each step, finding min degree among alive nodes.
    # degree[i] = sum(conflict[i, alive])
    # This is O(N) per step if we sum the row.
    # Total O(N^2) = 27M ops. Very fast in NumPy.
    
    while np.any(alive):
        if time.time() > deadline - 5:
            break
            
        # Calculate degrees for alive nodes
        # conflict is bool. alive is bool.
        # We want sum of conflict[i, :] where alive is True.
        # Use matrix multiplication or sum.
        # degrees = conflict @ alive
        # This is efficient.
        degrees = conflict @ alive
        
        # Mask out dead nodes to have high degree so they aren't picked
        # Or just find min among alive.
        degrees[~alive] = np.inf
        
        # Pick min degree
        min_idx = np.argmin(degrees)
        
        if not alive[min_idx]:
            break
            
        selected.append(min_idx)
        
        # Remove min_idx and its neighbors
        # Neighbors are where conflict[min_idx] is True.
        # We also remove min_idx itself.
        # So we set alive to False for min_idx and all j where conflict[min_idx, j] is True.
        
        # Update alive
        # alive = alive & ~conflict[min_idx]
        # But conflict[min_idx] includes min_idx itself (True).
        # So this works.
        
        alive = alive & ~conflict[min_idx]
        
    # Construct result
    if len(selected) > 0:
        result = V[selected]
    else:
        # Fallback
        result = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return result

# EVOLVE-BLOCK-END
