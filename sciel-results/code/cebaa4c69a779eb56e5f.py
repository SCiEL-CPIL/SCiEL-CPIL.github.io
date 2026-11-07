"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        """
        Solve clique partition for a split graph with:
        - K = clique on vertices 0..k-1
        - I = independent set on vertices k..n-1
        - Cross edges with probability p
        
        Strategy:
        1. Generate the graph
        2. Find an independent set vertex v with maximum degree in K
        3. Let S = N(v) (neighbors of v in K). Since K is a clique, {v} ∪ S is a clique.
        4. Create one big clique C = {v} ∪ S
        5. All other edges not in C become individual edge-cliques of size 2
        6. This minimizes the number of cliques
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        # cross_adj[v] = set of K vertices connected to v
        cross_adj = [set() for _ in range(n)]
        for (u, v) in edges:
            if u < k and v >= k:
                cross_adj[v].add(u)
                cross_adj[u].add(v)
            elif u >= k and v < k:
                cross_adj[u].add(v)
                cross_adj[v].add(u)
        
        # Find I vertex with max degree
        best_v = -1
        max_deg = -1
        for v in range(k, n):
            deg = len(cross_adj[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            # No cross edges, just return K as one clique if k >= 2
            if k >= 2:
                return [list(range(k))]
            else:
                return []
        
        # Create the big clique
        S = sorted(cross_adj[best_v])
        big_clique = [best_v] + S
        
        # Mark edges covered by big_clique
        # Edges in big_clique:
        # 1. (u, w) for u, w in S (all K-K edges within S)
        # 2. (u, best_v) for u in S (all cross edges from best_v)
        covered = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                u, w = S[i], S[j]
                if u > w:
                    u, w = w, u
                covered.add((u, w))
        for u in S:
            covered.add((min(u, best_v), max(u, best_v)))
        
        # Create list of cliques
        cliques = [big_clique]
        
        # Add all other edges as edge-cliques
        for (u, v) in edges:
            if (u, v) not in covered:
                cliques.append([u, v])
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > DEADLINE - 5:
            break
        cliques = solve_one(*spec)
        results.append(cliques)
    
    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
