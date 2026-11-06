def construct():
    import random
    from math import gcd

    n = 128
    elements = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    def is_bad(a, b, c):
        return lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]

    # 2. Precompute bad pairs for each x
    # bad_pairs_for_x[x] is a list of tuples (a, b) such that (a, b, x) is a bad triple
    bad_pairs_for_x = [[] for _ in range(n + 1)]
    all_bad_triples = set()
    
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x: continue
            for b in range(a + 1, n + 1):
                if b == x: continue
                if is_bad(a, b, x):
                    bad_pairs_for_x[x].append((a, b))
                    all_bad_triples.add((a, b, x))
    
    # Convert all_bad_triples to a list for iteration
    all_bad_triples_list = list(all_bad_triples)
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for a, b in bad_pairs_for_x[x]:
            if a in S and b in S:
                return False
        return True

    def find_bad_triple_in_S(S):
        """Find any bad triple fully contained in S."""
        for a, b, c in all_bad_triples_list:
            if a in S and b in S and c in S:
                return (a, b, c)
        return None

    def repair(S):
        """Remove elements until S is valid."""
        S = set(S)
        while True:
            triple = find_bad_triple_in_S(S)
            if triple is None:
                break
            a, b, c = triple
            # Heuristic: remove the element involved in the most active bad triples
            counts = {a: 0, b: 0, c: 0}
            for ta, tb, tc in all_bad_triples_list:
                if ta in S and tb in S and tc in S:
                    if a in (ta, tb, tc): counts[a] += 1
                    if b in (ta, tb, tc): counts[b] += 1
                    if c in (ta, tb, tc): counts[c] += 1
            
            # Pick the one with max count, tie-break randomly
            candidates = [a, b, c]
            max_count = max(counts.values())
            top_candidates = [v for v in candidates if counts[v] == max_count]
            to_remove = random.choice(top_candidates)
            S.remove(to_remove)
        return S

    def greedy_add(S):
        """Add elements to S if possible."""
        S = set(S)
        complement = set(elements) - S
        changed = True
        while changed:
            changed = False
            for x in sorted(complement):
                if can_add(x, S):
                    S.add(x)
                    complement.remove(x)
                    changed = True
        return S

    def local_search(S, iterations=50):
        """Perform local search with swaps."""
        S = set(S)
        best_S = S.copy()
        best_size = len(S)
        
        for _ in range(iterations):
            improved = False
            # Step 1: Greedy Add
            S = greedy_add(S)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                improved = True
                
            # Step 2: If no improvement from add, try a random swap
            if not improved:
                if S and (set(elements) - S):
                    y = random.choice(list(S))
                    x = random.choice(list(set(elements) - S))
                    S_temp = S - {y}
                    if can_add(x, S_temp):
                        S = S_temp | {x}
                        S = greedy_add(S)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.copy()
                            improved = True
                            
            if not improved:
                # Perturb: remove a random element to break local minimum
                if len(S) > 1:
                    y = random.choice(list(S))
                    S.remove(y)
                    S = greedy_add(S)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        
        return best_S

    # Main Execution
    best_set = []
    
    # Strategy 1: Start with all elements, repair, then improve
    S_init = set(elements)
    S_valid = repair(S_init)
    S_final = local_search(S_valid, iterations=50)
    if len(S_final) > len(best_set):
        best_set = S_final
        
    # Strategy 2: Start with odd numbers (heuristic), repair, improve
    S_odd = set(x for x in elements if x % 2 != 0)
    S_valid_odd = repair(S_odd)
    S_final_odd = local_search(S_valid_odd, iterations=50)
    if len(S_final_odd) > len(best_set):
        best_set = S_final_odd
        
    # Strategy 3: Random restarts
    rng = random.Random(123)
    for _ in range(20):
        S_rand = set(rng.sample(elements, rng.randint(40, 80)))
        S_valid_rand = repair(S_rand)
        S_final_rand = local_search(S_valid_rand, iterations=30)
        if len(S_final_rand) > len(best_set):
            best_set = S_final_rand
            
    if not best_set:
        best_set = [1]
        
    return list(best_set)
