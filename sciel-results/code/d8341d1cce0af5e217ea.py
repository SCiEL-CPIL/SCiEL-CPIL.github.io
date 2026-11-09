"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency: for each v in I (k..n-1), find S(v) = {u in C : edge (u,v)}
        # C is 0..k-1, I is k..n-1
        # C-C edges: one clique = list(range(k)) if k >= 2
        # For each v in I, S(v) = [u for u in range(k) if rng.random() < p]
        # But we need to generate in the same order as the seed to match the random sequence.
        # Seed generates: first all C-C edges (no randomness), then for each v in I (v from k to n-1),
        # for each u in C (u from 0 to k-1), rng.random() < p
        # So the random calls are in order: v=k, u=0; v=k, u=1; ...; v=k, u=k-1; v=k+1, u=0; ...
        
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        for v in range(k, n):
            S = []
            for u in range(k):
                if rng.random() < p:
                    S.append(u)
            if len(S) >= 1:
                # Clique: v plus all u in S
                clique = S + [v]
                # Ensure at least 2 vertices (guaranteed since len(S)>=1, so len>=2)
                cliques.append(clique)
        
        return cliques

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = instance(*spec)
        # Convert to ensure Python int types
        converted = []
        for c in cliques:
            converted.append([int(x) for x in c])
        results.append(converted)
    
    # If we didn't finish all, pad with empty? No, we must return 3 lists.
    # But the algorithm is O(n*k) which is tiny, so this won't happen.
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
