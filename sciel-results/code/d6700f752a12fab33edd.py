"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import random
    from collections import defaultdict

    N = 128
    numbers = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # Build 3-uniform hypergraph: hyperedges are bad triples (a,b,c) where
    # gcd(a,b) == gcd(a,c) == gcd(b,c)
    # For efficient membership testing, store for each pair (x,y) the set of z
    # that form a bad triple with x and y.
    pair_to_z = defaultdict(set)  # (min(x,y), max(x,y)) -> set of z
    
    # Also compute hypergraph degree for each vertex
    deg = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d_ab = gcd_matrix[a][b]
            for c in range(b + 1, N + 1):
                if gcd_matrix[a][c] == d_ab and gcd_matrix[b][c] == d_ab:
                    # Bad triple (a, b, c)
                    pair_to_z[(a, b)].add(c)
                    pair_to_z[(a, c)].add(b)
                    pair_to_z[(b, c)].add(a)
                    deg[a] += 1
                    deg[b] += 1
                    deg[c] += 1

    def can_add(v, S):
        """Check if v can be added to set S without forming a bad triple."""
        # For every pair (u, w) in S, check if (v, u, w) is a bad triple
        # Equivalent: for each u in S, for each w in S with w > u,
        #   check if gcd(v,u) == gcd(v,w) == gcd(u,w)
        # More efficient: for each u in S, let g_vu = gcd(v,u).
        #   For each w in S, if gcd(v,w) == g_vu and gcd(u,w) == g_vu, bad.
        sl = list(S)
        n = len(sl)
        for i in range(n):
            u = sl[i]
            g_vu = gcd_matrix[v][u]
            for j in range(i + 1, n):
                w = sl[j]
                if gcd_matrix[v][w] == g_vu and gcd_matrix[u][w] == g_vu:
                    return False
        return True

    def greedy_weighted(seed):
        """Weighted greedy: prioritize vertices with lowest hypergraph degree."""
        rng = random.Random(seed)
        # Create scored list: primary key = degree, secondary = random noise
        # to break ties and add diversity
        scored = [(deg[v] + rng.random() * 0.3, v) for v in range(1, N + 1)]
        scored.sort()
        
        S = set()
        for _, v in scored:
            if can_add(v, S):
                S.add(v)
        return S

    def local_search(S, max_iters=100):
        """Local search: try additions, then 1-for-k exchanges, then 1-for-1 swaps."""
        S = set(S)
        
        for _ in range(max_iters):
            improved = False
            
            # Phase 1: Try to add any vertex not in S
            for v in range(1, N + 1):
                if v not in S and can_add(v, S):
                    S.add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-2 or 1-for-3 exchange
            # Remove one vertex u, then greedily add as many as possible
            best_swap = None
            best_size = len(S)
            
            for u in list(S):
                S_temp = S - {u}
                # Try to add vertices in order of increasing degree
                addable = [v for v in range(1, N + 1) if v not in S_temp and v != u]
                addable.sort(key=lambda x: deg[x])
                
                S_greedy = set(S_temp)
                for v in addable:
                    if can_add(v, S_greedy):
                        S_greedy.add(v)
                
                if len(S_greedy) > best_size:
                    best_size = len(S_greedy)
                    best_swap = S_greedy
            
            if best_swap is not None:
                S = best_swap
                improved = True
                continue
            
            # Phase 3: Try 1-for-1 swap to escape local optimum
            # Replace high-degree u with low-degree v if it enables future gains
            swapped = False
            S_sorted = sorted(S, key=lambda x: -deg[x])
            candidates = sorted(range(1, N + 1), key=lambda x: deg[x])
            
            for u in S_sorted:
                for v in candidates:
                    if v in S:
                        continue
                    if deg[v] >= deg[u]:
                        break
                    # Check if replacing u with v is valid
                    S_test = S - {u}
                    if can_add(v, S_test):
                        S = S_test
                        S.add(v)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                break
        
        return S

    # Run multiple randomized restarts
    best_S = set()
    best_size = 0
    num_restarts = 50
    
    for i in range(num_restarts):
        S = greedy_weighted(seed=i)
        S = local_search(S, max_iters=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
