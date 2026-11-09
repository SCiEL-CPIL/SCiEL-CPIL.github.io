"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct a triangle-free graph on Z_n with alpha < 10.
    # Use Cayley graphs with sum-free connection sets.
    #
    # Key insight: If S is a sum-free subset of Z_n, then the Cayley graph
    # with connection set S is triangle-free (since if a~b and b~c, then 
    # b-a and c-b are in S, so c-a = (b-a)+(c-b) is in S+S, which is disjoint from S).
    #
    # For Z_40: S = {14, 15, ..., 26} is sum-free and symmetric.
    # This gives a triangle-free graph with alpha <= 3 (definitely < 10).
    #
    # Let's verify: n=40, S={14,...,26}
    # S+S = {28,...,52} mod 40 = {28,...,39, 0,...,12}
    # S ∩ (S+S) = {14,...,26} ∩ {28,...,39, 0,...,12} = ∅
    # So S is sum-free, hence triangle-free.
    #
    # Independent sets: two vertices u,v are non-adjacent iff (v-u) mod 40 ∉ S,
    # i.e., (v-u) mod 40 ∈ {0, 1, ..., 13, 27, ..., 39}.
    # The complement of S has 27 elements. But we need to bound alpha.
    #
    # Actually, let's use a simpler construction. The Andrasfai graph And(3) on 39 vertices
    # is known to be triangle-free with alpha = 13. But we need alpha < 10.
    #
    # Let's use the Clebsch graph or another known construction.
    # Actually, the best known is n=39 from the survey.
    #
    # Let me try a different approach: use a random search or a known construction.
    #
    # The (3,10)-Ramsey problem: R(3,10) <= 41, so we can have at most 40 vertices.
    # The largest known is 39.
    #
    # Let's construct a graph on 39 vertices using a known method.
    # One approach: take a 7-regular triangle-free graph on 39 vertices with alpha < 10.
    #
    # Actually, let's use the construction from the literature. The known 39-vertex graph
    # is based on the Paley graph or a similar construction.
    #
    # Let me try: vertices are Z_39. Connect i~j if (j-i) mod 39 is a quadratic residue.
    # But Z_39 is not a prime, so Paley doesn't directly apply.
    #
    # Alternative: use a random 7-regular graph and check. But that's not deterministic.
    #
    # Let's use a simpler construction. The Andrásfai graph And(3) has 3n vertices.
    # For n=13, that's 39 vertices. It's triangle-free.
    # The independence number of And(3) is n = 13. That's too large (need < 10).
    #
    # So And(3) on 39 vertices doesn't work directly.
    #
    # Let me think differently. We need a triangle-free graph on 39 vertices with alpha <= 9.
    # By the Ramsey bound, R(3,10) <= 41, so such a graph should exist.
    #
    # Let's try a random search with a time limit.
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 100  # Leave 20s safety margin
    
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def max_independent_set(n, edges, limit=10):
        # Branch and bound to find if there's an independent set of size >= limit
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Use a simple DFS with pruning
        def dfs(candidates, size):
            if size >= limit:
                return True
            if not candidates:
                return False
            # Pruning: if size + len(candidates) < limit, can't reach limit
            if size + len(candidates) < limit:
                return False
            v = candidates[0]
            # Include v
            new_candidates = [u for u in candidates[1:] if not adj[v][u]]
            if dfs(new_candidates, size + 1):
                return True
            # Exclude v
            if dfs(candidates[1:], size):
                return True
            return False
        
        return dfs(list(range(n)), 0)
    
    # Try random graphs
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i+1, 26) if (j-i) % 26 % 3 == 1]
    
    # Seed the random number generator for reproducibility
    random.seed(42)
    
    # Try to find a valid graph on 39 vertices
    for n in range(39, 38, -1):  # Try 39 first, then 38
        if time.time() > deadline:
            break
        
        max_attempts = 500 if n == 39 else 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate a random graph with expected degree ~7
            # For n=39, average degree ~7 means ~39*7/2 = 136.5 edges
            num_edges = n * 7 // 2
            
            edges = []
            edge_set = set()
            attempts = 0
            while len(edges) < num_edges and attempts < num_edges * 10:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u < v:
                    if (u, v) not in edge_set:
                        edge_set.add((u, v))
                        edges.append([u, v])
                attempts += 1
            
            if len(edges) != num_edges:
                continue
            
            # Check triangle-free
            if not is_triangle_free(n, edges):
                continue
            
            # Check independence number < 10
            if max_independent_set(n, edges, 10):
                continue
            
            # Found a valid graph!
            return {'n': n, 'edges': edges}
    
    # If we didn't find one, return the seed graph
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
