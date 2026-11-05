"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a maximum subset of [1,128] with no three distinct elements
    whose three pairwise LCMs are all equal.
    
    Uses branch-and-bound on the 3-uniform hypergraph of bad triples,
    maintaining a set of active pairs for fast conflict detection.
    """
    from math import gcd
    from collections import defaultdict
    import sys
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We group by LCM value to find them efficiently.
    pairs_by_lcm = defaultdict(list)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm(a, b)
            pairs_by_lcm[L].append((a, b))
            
    bad_triples = set()
    for L, pairs in pairs_by_lcm.items():
        # Build adjacency for this specific LCM value
        # If (a,b) and (a,c) are in pairs, and lcm(b,c) == L, then (a,b,c) is bad.
        # Actually, if lcm(a,b)=L and lcm(a,c)=L, it doesn't guarantee lcm(b,c)=L.
        # Example: a=2, b=3, c=5. Lcm(2,3)=6, Lcm(2,5)=10. Not same L.
        # So we only look at pairs with the SAME LCM L.
        # If lcm(a,b)=L and lcm(a,c)=L, then b and c must be divisors of L.
        # Does lcm(b,c)=L?
        # Let L = p^k * ...
        # a has p^alpha, b has p^beta, c has p^gamma.
        # max(alpha, beta) = k, max(alpha, gamma) = k.
        # This implies beta=k or alpha=k, and gamma=k or alpha=k.
        # If alpha < k, then beta=k and gamma=k, so max(beta, gamma)=k.
        # If alpha = k, then max(beta, gamma) could be < k? No, beta and gamma are exponents of b,c.
        # Wait, if alpha=k, then lcm(a,b) has p^k. lcm(a,c) has p^k.
        # lcm(b,c) has p^max(beta, gamma).
        # It is possible that beta < k and gamma < k?
        # If beta < k and alpha = k, then lcm(a,b) has p^k. Correct.
        # If gamma < k and alpha = k, then lcm(a,c) has p^k. Correct.
        # Then lcm(b,c) has p^max(beta, gamma) < k.
        # So lcm(b,c) != L.
        # Thus, we MUST check lcm(b,c) == L.
        
        # For each pair (a, b) in pairs, check other pairs (a, c) or (b, c)
        # This is O(M^2) where M is number of pairs for L.
        # Sum of M^2 might be large, but N=128 is small.
        
        # Optimization: For each node 'a' in this LCM graph, look at its neighbors.
        # If b and c are neighbors of a, check if lcm(b,c)==L.
        adj = defaultdict(list)
        for a, b in pairs:
            adj[a].append(b)
            adj[b].append(a)
            
        for a in adj:
            neighbors = adj[a]
            for i in range(len(neighbors)):
                b = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    c = neighbors[j]
                    if lcm(b, c) == L:
                        triple = tuple(sorted((a, b, c)))
                        bad_triples.add(triple)

    # 2. Precompute conflict pairs for each vertex
    # conflict_pairs[v] = set of frozensets {u, w} such that {u, v, w} is a bad triple
    conflict_pairs = defaultdict(set)
    for a, b, c in bad_triples:
        conflict_pairs[a].add(frozenset((b, c)))
        conflict_pairs[b].add(frozenset((a, c)))
        conflict_pairs[c].add(frozenset((a, b)))
        
    # 3. Vertex Ordering
    # Order vertices by the number of conflicts they have (descending) to prune earlier
    degrees = {v: len(conflict_pairs[v]) for v in range(1, n + 1)}
    ordered_vertices = sorted(range(1, n + 1), key=lambda x: degrees[x], reverse=True)
    # Map index in ordered list to value
    # We will iterate through indices 0..n-1
    
    best_size = 0
    best_set = []
    
    # State for B&B:
    # selected: set of values
    # active_pairs: set of frozensets {u, w} where u, w in selected
    # idx: current index in ordered_vertices
    
    sys.setrecursionlimit(10000)
    
    def solve(idx, selected, active_pairs, current_size):
        nonlocal best_size, best_set
        
        # Upper bound: current_size + remaining vertices
        if current_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = list(selected)
            return
            
        v = ordered_vertices[idx]
        
        # Option 1: Include v
        # Check if v conflicts with any active pair
        can_include = True
        for pair in conflict_pairs[v]:
            if pair in active_pairs:
                can_include = False
                break
                
        if can_include:
            # Add v
            new_selected = selected | {v}
            new_active_pairs = set(active_pairs)
            # Add pairs (v, u) for all u in selected
            for u in selected:
                new_active_pairs.add(frozenset((v, u)))
                
            solve(idx + 1, new_selected, new_active_pairs, current_size + 1)
            
        # Option 2: Exclude v
        solve(idx + 1, selected, active_pairs, current_size)
        
    # Initial call
    solve(0, set(), set(), 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
