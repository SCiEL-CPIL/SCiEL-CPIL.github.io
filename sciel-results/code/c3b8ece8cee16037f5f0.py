"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import random
    import time

    N = 128
    numbers = list(range(1, N + 1))

    # 1. Precompute GCD matrix and Bad Triples (Hyperedges)
    # gcd_matrix[i][j] = gcd(i, j)
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # Hyperedges: sets of 3 distinct vertices {a, b, c} such that gcd(a,b)==gcd(a,c)==gcd(b,c)
    # We store these as tuples (a, b, c) with a < b < c for uniqueness.
    bad_triples = []
    # Degree of each vertex in the hypergraph (number of bad triples containing it)
    degree = [0] * (N + 1)
    
    # Optimization: Precompute for each pair (i, j) the gcd.
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c) = d.
    # This implies a, b, c are all multiples of d, and after dividing by d, they are pairwise coprime?
    # No, gcd(a/d, b/d) = 1, gcd(a/d, c/d) = 1, gcd(b/d, c/d) = 1.
    # So a/d, b/d, c/d must be pairwise coprime.
    
    # Let's just iterate all triples to be safe and simple, N=128 is small.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d_ab = gcd_matrix[a][b]
            for c in range(b + 1, N + 1):
                if gcd_matrix[a][c] == d_ab and gcd_matrix[b][c] == d_ab:
                    bad_triples.append((a, b, c))
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1

    # 2. Helper function to check if a vertex can be added to a set
    # We need to check if adding v to S creates any bad triple.
    # A bad triple involving v must be (v, u, w) where u, w in S.
    # Condition: gcd(v, u) == gcd(v, w) == gcd(u, w).
    
    def can_add(v, S, gcd_matrix):
        # Iterate over pairs in S
        # Optimization: If S is large, this is O(|S|^2). 
        # We can optimize by precomputing conflicts, but for N=128 and |S| < 128, it's fine.
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            u = s_list[i]
            g_vu = gcd_matrix[v][u]
            for j in range(i + 1, n):
                w = s_list[j]
                if gcd_matrix[v][w] == g_vu:
                    if gcd_matrix[u][w] == g_vu:
                        return False
        return True

    # 3. Weighted Greedy Algorithm
    # Prioritize vertices with lowest hypergraph degree.
    # Ties broken randomly to allow for multiple restarts or variations.
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        # Sort vertices by degree, then by random noise for tie-breaking
        # Lower degree = safer = pick earlier
        vertices = list(range(1, N + 1))
        # Add a small random factor to degree to break ties and explore different orderings
        # If we use exact degree, we might get stuck in a local optimum specific to that order.
        # The directive says "prioritizes vertices with the lowest hypergraph degree".
        # I will use (degree[v], rng.random()) as the sort key.
        
        order = sorted(vertices, key=lambda v: (degree[v], rng.random()))
        
        S = set()
        for v in order:
            if can_add(v, S, gcd_matrix):
                S.add(v)
        return S

    # 4. Local Search
    # Try to improve the set S by:
    # a) Adding any valid vertex.
    # b) Swapping a vertex u in S with a vertex v not in S if it allows adding more vertices or just increases size (which swap doesn't do, but enables future additions).
    # c) Removing u and adding multiple vertices.
    
    def local_search(S, max_iters=100):
        S = set(S)
        for _ in range(max_iters):
            improved = False
            
            # Step 1: Try to add any valid vertex
            for v in range(1, N + 1):
                if v not in S:
                    if can_add(v, S, gcd_matrix):
                        S.add(v)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Step 2: Try 1-for-k exchanges
            # For each u in S, remove it and see how many vertices can be added.
            # If count_added > 1, we gain size.
            best_gain = 0
            best_move = None # (u, new_S)
            
            for u in list(S):
                S_temp = S - {u}
                addable = []
                for v in range(1, N + 1):
                    if v not in S_temp:
                        if can_add(v, S_temp, gcd_matrix):
                            addable.append(v)
                
                if len(addable) >= 2:
                    # Greedily add them to see net gain
                    S_greedy = set(S_temp)
                    added = 0
                    for v in addable:
                        if can_add(v, S_greedy, gcd_matrix):
                            S_greedy.add(v)
                            added += 1
                    if added > 1:
                        gain = added - 1
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (u, S_greedy)
            
            if best_move:
                u, S_new = best_move
                S = S_new
                improved = True
                continue
            
            # Step 3: Try 1-for-1 swap to enable future improvements (neutral move)
            # This is harder to justify without lookahead. 
            # Instead, let's try a "kick": remove a random vertex, then re-greedy fill.
            # Or just break if no improvement.
            
            # Let's try a simple kick: remove one vertex, then try to add others.
            # We already did 1-for-2. 
            # Let's try 1-for-1 where the new vertex has lower degree, hoping it's "safer".
            # Actually, the greedy step already prioritized low degree. 
            # Let's just break if no size-increasing move found.
            # To avoid getting stuck, we can restart with a different seed if we hit a plateau.
            
            break
            
        return S

    # 5. Multiple Restarts
    # Run greedy + local search multiple times with different seeds.
    
    best_S = set()
    best_size = 0
    num_restarts = 50 # Increase restarts to find better solutions
    
    for i in range(num_restarts):
        S = greedy_construct(seed=i)
        S = local_search(S, max_iters=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Final attempt: Try to add any remaining vertices to best_S
    for v in range(1, N + 1):
        if v not in best_S:
            if can_add(v, best_S, gcd_matrix):
                best_S.add(v)

    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
