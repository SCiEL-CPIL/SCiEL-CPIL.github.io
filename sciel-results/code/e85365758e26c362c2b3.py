"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

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
    # K2,2,2 is the complement of a perfect matching on 6 vertices.
    # For a 6-subset, let E_c be the set of edges of color c.
    # If |E_c| == 12, then the other 3 edges (color 1-c) form a perfect matching?
    # Yes, if the 3 edges of color 1-c form a perfect matching, then it's a K2,2,2 in color c.
    
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

def _local_search(adj, n, deadline, rng):
    """
    Perform local search to reduce violations.
    Returns the best adj found and the violation count.
    """
    current_violations = _count_k222_violations(adj, n)
    
    if current_violations == 0:
        return adj, 0
    
    # Precompute edge list for faster access
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Use a simple first-improvement or best-improvement heuristic
    # For speed, we'll do random restarts with simulated annealing-like acceptance
    
    best_adj = adj[:]
    best_violations = current_violations
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random edge
        idx = rng.randint(0, len(edges) - 1)
        u, v = edges[idx]
        
        # Flip the edge
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_violations = _count_k222_violations(adj, n)
        
        delta = new_violations - current_violations
        
        if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_adj = adj[:]
                if best_violations == 0:
                    break
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        T *= cooling
        
    return best_adj, best_violations

def _try_extend(adj, n, deadline, rng):
    """
    Try to extend a valid coloring of K_n to K_{n+1} by adding a new vertex.
    Returns (new_adj, new_n) if successful, else (None, n).
    """
    # We need to find a vector of length n (colors for edges from new vertex to old vertices)
    # such that both color classes remain K2,2,2-free.
    
    # This is a constraint satisfaction problem.
    # We can use backtracking with pruning.
    
    # For each new vertex v_new, we assign colors c_0, c_1, ..., c_{n-1} to edges (v_new, i).
    
    # Pruning condition:
    # If we add an edge of color c to v_new, we must ensure that the neighborhood of v_new in color c
    # does not create a K2,2,2 with existing edges.
    
    # Specifically, if v_new is connected to vertices S in color c, then the subgraph induced by S
    # must not contain a K2,2,2 that together with v_new forms a K2,2,2?
    # Actually, K2,2,2 has 6 vertices. If v_new is one of them, the other 5 must be in S.
    # But K2,2,2 is tripartite. If v_new is in one part, the other 2 parts must be in S.
    # So S must not contain a K2,2 (i.e., a complete bipartite graph with parts of size 2 and 2)
    # in color c? No, that's not quite right.
    
    # Let's think differently.
    # A K2,2,2 in the new graph either:
    # 1. Does not involve v_new. (Already checked, since the old graph is valid.)
    # 2. Involves v_new.
    
    # Case 2: v_new is in the K2,2,2.
    # The K2,2,2 has parts of size 2. v_new is in one part, say A.
    # The other parts B and C are subsets of the old vertices.
    # |A| = 2, so there is one other vertex a in A.
    # |B| = 2, |C| = 2.
    # All edges between A and B, A and C, B and C are of color c.
    # So, v_new must be connected to a, B, and C in color c.
    # And a must be connected to B and C in color c.
    # And B and C must be connected in color c.
    
    # So, for a fixed color c, if we choose the set S of vertices connected to v_new in color c,
    # then S must not contain a vertex a such that the neighborhood of a in color c (within the old graph)
    # contains a K2,2? No.
    
    # Let's use a simpler pruning:
    # For each pair of vertices (i, j) in the old graph that are connected in color c,
    # if both i and j are in S, and there exists a K2,2,2 involving v_new, i, j...
    
    # This is getting complex. Let's just use brute force for small n.
    # For n=29, we have 2^29 possibilities. That's 536 million. Too many.
    
    # We need a smarter approach.
    # Let's use the fact that the Paley graph is symmetric.
    # Maybe we can try specific extensions.
    
    # For now, let's just return failure.
    return None, n

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
    adj, current_violations = _local_search(adj, n, deadline, rng)
    
    if current_violations == 0:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    
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
