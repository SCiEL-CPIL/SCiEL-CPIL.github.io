"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _count_k222_violations(adj, n):
    """
    Count the number of K2,2,2 subgraphs in the graph defined by adj.
    Returns the count of monochromatic K2,2,2s.
    """
    if n < 6:
        return 0
    
    count = 0
    # K2,2,2 requires 6 vertices.
    # We iterate over pairs of edges (a,b) and (c,d) that are disjoint? No, 
    # K2,2,2 is a complete tripartite graph with parts of size 2.
    # Vertices: {a,b}, {c,d}, {e,f}.
    # Edges: all pairs between parts.
    # A K2,2,2 is determined by 6 vertices.
    # To count efficiently, we can look for structures.
    # However, for n=29, the number of 6-subsets is C(29,6) ~ 475k.
    # Checking each for K2,2,2 structure is feasible.
    
    # A set of 6 vertices {v0..v5} forms a K2,2,2 if there exists a partition into 3 pairs
    # such that all 12 cross-edges are present.
    # There are 15 ways to partition 6 items into 3 pairs.
    # We can check if the induced subgraph contains a K2,2,2.
    
    # Optimization: Precompute adjacencies.
    # For each 6-subset, check if it contains K2,2,2.
    # This might be slow in pure Python if done naively.
    # Let's try a smarter check.
    
    # Alternative:
    # A K2,2,2 contains a C4.
    # But checking all 6-subsets is the most robust way to count exactly.
    
    # Let's optimize the 6-subset check.
    # Generate all 6-combinations.
    # For each combo, check the 15 pairings.
    
    # Since n is small (<=31), we can use bitmasks.
    # But iterating 6-combinations in Python is slow.
    # C(31, 6) = 736,281.
    # 736k iterations * 15 checks * bit ops.
    # 736k * 15 = 11M operations. This is borderline for 120s in Python.
    # We need to be careful.
    
    # Let's use a different strategy for counting:
    # Count specific configurations.
    # Actually, for local search, we don't need the *exact* count of all K2,2,2s.
    # We can use a proxy metric or a faster check.
    # However, the exact count is the best signal.
    
    # Let's try to speed up the 6-subset check.
    # We can precompute for each pair of vertices (u,v) the common neighbors.
    # But K2,2,2 involves 6 vertices.
    
    # Let's stick to the 6-subset enumeration but optimize the inner loop.
    # We will iterate over all combinations of 6 vertices.
    # For each combination, we check if the induced subgraph is a K2,2,2.
    # The induced subgraph on 6 vertices has 15 edges.
    # K2,2,2 has 12 edges.
    # So the induced subgraph must have exactly 12 edges of color 0 (or 1)
    # AND those edges must form a K2,2,2.
    # Actually, if the induced subgraph has 12 edges of color 0, it is a K2,2,2
    # IF AND ONLY IF the complement of those edges in the 6-vertex graph is a matching of size 3?
    # No, K2,2,2 is the complement of a perfect matching on 6 vertices.
    # Yes! K_{2,2,2} is isomorphic to the complement of a 1-factor (perfect matching) on 6 vertices.
    # So, a 6-subset contains a monochromatic K2,2,2 in color c if and only if
    # the edges of color c in the induced subgraph form the complement of a perfect matching.
    # This means the edges of color c are all edges EXCEPT the 3 matching edges.
    # So, for a 6-subset, let E_c be the set of edges of color c.
    # If |E_c| == 12, then the other 3 edges (color 1-c) form a perfect matching?
    # Not necessarily. The 3 non-c edges could be a path or triangle, etc.
    # But if |E_c| == 12, the 3 non-c edges must be a matching for it to be a K2,2,2.
    # Why? Because in K2,2,2, the non-edges are exactly the 3 pairs within the 3 parts.
    # These 3 pairs form a perfect matching.
    # So, condition:
    # 1. Number of edges of color c in the 6-subset is 12.
    # 2. The 3 edges of color 1-c form a perfect matching (no shared vertices).
    
    # This is much faster to check!
    # For each 6-subset:
    #   count edges of color 0.
    #   if count == 12: check if color 1 edges form a matching.
    #   if count == 3: check if color 0 edges form a matching. (This covers color 1 K2,2,2)
    #   if count == 0 or 15: no.
    #   Wait, if count(color 0) == 12, then count(color 1) == 3.
    #   We just need to check if the 3 color 1 edges form a matching.
    #   If yes, it's a K2,2,2 in color 0.
    #   If count(color 0) == 3, then count(color 1) == 12.
    #   We check if the 3 color 0 edges form a matching.
    #   If yes, it's a K2,2,2 in color 1.
    
    # So we only need to check the 3 edges of the minority color.
    
    # Implementation:
    # Iterate all 6-combinations.
    # For each, count edges of color 0.
    # If count is 3 or 12, check the matching property of the 3 edges of that color.
    
    # To make this fast in Python, we might need to prune or use numpy.
    # But 736k combinations is okay if the inner loop is tight.
    
    # Let's implement this.
    
    # Pre-extract edges? No, adj bitmasks are good.
    
    # To speed up, we can skip 6-subsets that don't have the right edge count quickly?
    # Summing 15 bits for each of 736k subsets is ~11M bit ops. Python might struggle.
    # 11M operations in Python loop is ~1-2 seconds.
    # We have 120s. We can afford ~50-100 full counts.
    # Local search might need more iterations.
    # We can optimize by only checking a subset of edges or using a heuristic count.
    # But let's try the exact count first. If it's too slow, we'll switch to a heuristic.
    
    # Actually, we can use a faster proxy:
    # Count the number of "bad" 6-sets is the goal.
    
    # Let's write the counter.
    
    import itertools
    
    total_violations = 0
    
    # Optimization: If n is large, this is slow.
    # For n=29, C(29,6) = 475020.
    # 475k * 15 checks is manageable.
    
    for combo in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = combo
        # Check edges
        # There are 15 pairs.
        # We need to count how many are color 0.
        # And identify the 3 edges if count is 3 or 12.
        
        c0_count = 0
        # We also need the indices of the 3 edges if count is 3 (for color 0) or 12 (for color 1).
        # It's easier to just collect the edges of color 0.
        # If count is 3, check if they form a matching.
        # If count is 12, the color 1 edges are the complement. Check if they form a matching.
        
        # Let's just count c0.
        # And store the 3 edges if it's 3, or the 3 non-edges if it's 12.
        
        # To avoid overhead, we can just count.
        # But we need the edges to check matching.
        
        # Let's collect edges of color 0.
        edges_0 = []
        
        # Pairs in 6-set
        # (0,1), (0,2), (0,3), (0,4), (0,5)
        # (1,2), (1,3), (1,4), (1,5)
        # (2,3), (2,4), (2,5)
        # (3,4), (3,5)
        # (4,5)
        
        # Helper to check edge
        def is_c0(u, v):
            return (adj[u] >> v) & 1
        
        # We can unroll this for speed.
        
        e01 = is_c0(v0, v1)
        e02 = is_c0(v0, v2)
        e03 = is_c0(v0, v3)
        e04 = is_c0(v0, v4)
        e05 = is_c0(v0, v5)
        
        e12 = is_c0(v1, v2)
        e13 = is_c0(v1, v3)
        e14 = is_c0(v1, v4)
        e15 = is_c0(v1, v5)
        
        e23 = is_c0(v2, v3)
        e24 = is_c0(v2, v4)
        e25 = is_c0(v2, v5)
        
        e34 = is_c0(v3, v4)
        e35 = is_c0(v3, v5)
        
        e45 = is_c0(v4, v5)
        
        c0_count = e01 + e02 + e03 + e04 + e05 + e12 + e13 + e14 + e15 + e23 + e24 + e25 + e34 + e35 + e45
        
        if c0_count == 3:
            # Check if the 3 edges of color 0 form a perfect matching.
            # The 3 edges must be disjoint.
            # We can check this by seeing if the 6 vertices are covered by 3 disjoint edges.
            # Since there are exactly 3 edges, they form a matching iff no two share a vertex.
            # We can check the degree of each vertex in the subgraph of color 0 edges.
            # Each vertex must have degree 1.
            
            # Degrees:
            # deg(v0) = e01+e02+e03+e04+e05
            # deg(v1) = e01+e12+e13+e14+e15
            # ...
            
            d0 = e01 + e02 + e03 + e04 + e05
            d1 = e01 + e12 + e13 + e14 + e15
            d2 = e02 + e12 + e23 + e24 + e25
            d3 = e03 + e13 + e23 + e34 + e35
            d4 = e04 + e14 + e24 + e34 + e45
            d5 = e05 + e15 + e25 + e35 + e45
            
            if d0==1 and d1==1 and d2==1 and d3==1 and d4==1 and d5==1:
                total_violations += 1
                
        elif c0_count == 12:
            # The 3 edges of color 1 form a perfect matching.
            # This is equivalent to the 3 non-edges of color 0 forming a matching.
            # The degrees in color 1 subgraph are 1 - degrees in color 0 subgraph? No.
            # deg_1(v) = 5 - deg_0(v).
            # We need deg_1(v) == 1 for all v.
            # So 5 - deg_0(v) == 1 => deg_0(v) == 4.
            
            d0 = e01 + e02 + e03 + e04 + e05
            d1 = e01 + e12 + e13 + e14 + e15
            d2 = e02 + e12 + e23 + e24 + e25
            d3 = e03 + e13 + e23 + e34 + e35
            d4 = e04 + e14 + e24 + e34 + e45
            d5 = e05 + e15 + e25 + e35 + e45
            
            if d0==4 and d1==4 and d2==4 and d3==4 and d4==4 and d5==4:
                total_violations += 1
                
    return total_violations

def _is_valid(adj, n):
    return _count_k222_violations(adj, n) == 0

def _build_paley(q):
    """Build Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 110
    rng = random.Random(42)
    
    # Start with Paley 29
    n = 29
    adj = _build_paley(n)
    
    # Check if it's already valid
    current_violations = _count_k222_violations(adj, n)
    
    if current_violations == 0:
        # It's valid! Return it.
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    
    # If not valid, try to fix it with local search
    # We try to flip edges to reduce violations.
    
    # To speed up, we can cache the violations?
    # Or just recompute.
    
    # Local Search Loop
    max_iterations = 10000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        # Or iterate through all edges?
        # Random is better for escaping local minima.
        
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_violations = _count_k222_violations(adj, n)
        
        if new_violations < current_violations:
            current_violations = new_violations
            if current_violations == 0:
                break
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # If we didn't reach 0, try a different starting point?
    # Or try n=31?
    # The task says maximize n.
    # If 29 is not fixed, maybe 29 is the max?
    # But we should try to extend or find a valid one.
    
    # If we failed to fix 29, we return the best we have (which might be invalid).
    # But the evaluator requires a valid coloring?
    # "A valid colouring of K_n certifies R(K2,2,2) > n."
    # "Neither colour class may contain K2,2,2..."
    # So the returned coloring MUST be valid.
    # If we can't make 29 valid, we should fall back to a smaller valid n.
    
    # Let's check if the current adj is valid.
    if _is_valid(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    
    # Fallback: Try smaller Paley graphs
    for q in [17, 13, 11, 5]:
        if time.time() > deadline:
            break
        adj_q = _build_paley(q)
        if _is_valid(adj_q, q):
            colors = []
            for i in range(q):
                for j in range(i + 1, q):
                    if (adj_q[i] >> j) & 1:
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': q, 'colors': colors}
            
    # Emergency fallback: n=1
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
