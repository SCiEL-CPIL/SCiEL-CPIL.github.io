"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 80  # safety margin before 90s kill
    
    N = 512
    
    # Build compatibility graph: edge between a,b if lcm(a,b) <= 512
    # lcm(a,b) = a*b // gcd(a,b)
    compat = [bytearray(N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= N:
                compat[a][b] = 1
                compat[b][a] = 1
    
    # Use greedy coloring + branch and bound for max clique
    # Start with all vertices, use a simple approach
    vertices = list(range(1, N + 1))
    
    # Simple greedy: sort by degree descending, add if compatible with all selected
    # Then local search to improve
    
    def try_greedy(order):
        selected = []
        for v in order:
            if all(compat[v][s] for s in selected):
                selected.append(v)
        return selected
    
    # Try multiple orderings
    import random
    best = []
    
    # Degree-based ordering
    degrees = [sum(compat[v][w] for w in range(1, N+1) if w != v) for v in range(1, N+1)]
    order1 = sorted(range(1, N+1), key=lambda v: -degrees[v-1])
    best = try_greedy(order1)
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(500):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        # Bias toward high-degree vertices
        order.sort(key=lambda v: -degrees[v-1] + rng.random() * 10)
        result = try_greedy(order)
        if len(result) > len(best):
            best = result
    
    # Local search: try to add removed vertices
    in_set = set(best)
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for v in range(1, N+1):
            if v not in in_set:
                if all(compat[v][s] for s in in_set):
                    in_set.add(v)
                    best = list(in_set)
                    improved = True
    
    return list(best)
# EVOLVE-BLOCK-END
