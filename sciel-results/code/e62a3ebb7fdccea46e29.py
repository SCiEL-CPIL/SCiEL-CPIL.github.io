def construct():
    import itertools
    
    n = 16
    all_points = list(range(n))
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Greedy: add triples that maintain linearity
    # Priority: triples that cover "hard" points
    triples = []
    used_pairs = set()
    
    def get_pair_set(triple):
        a, b, c = triple
        return frozenset({(a,b), (a,c), (b,c)})
    
    # Sort triples by some heuristic - maybe by how many points they cover
    # that are currently under-covered
    point_coverage = [0] * n
    
    # Try multiple random orderings and pick best
    import random
    
    best_triples = None
    best_indep = n + 1
    
    for attempt in range(50):
        random.seed(attempt)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        cur_triples = []
        cur_used_pairs = set()
        cur_point_coverage = [0] * n
        
        for t in shuffled:
            a, b, c = t
            pairs = {(a,b), (a,c), (b,c)}
            # Check linearity
            if any(p in cur_used_pairs for p in pairs):
                continue
            # Add
            cur_triples.append(list(t))
            cur_used_pairs.update(pairs)
            cur_point_coverage[a] += 1
            cur_point_coverage[b] += 1
            cur_point_coverage[c] += 1
        
        # Compute independence number
        # Use branch and bound or subset enumeration
        # For 16 points, we can enumerate subsets of size k from largest down
        from math import comb
        
        def has_triple_in_subset(subset):
            subset_set = set(subset)
            for t in cur_triples:
                if all(p in subset_set for p in t):
                    return True
            return False
        
        # Binary search or linear search from top
        indep = n
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                if not has_triple_in_subset(subset):
                    found = True
                    break
            if found:
                indep = size
                break
            else:
                indep = size - 1
                break
        
        if indep < best_indep:
            best_indep = indep
            best_triples = cur_triples[:]
        
        if best_indep <= 7:  # Good enough
            break
    
    return best_triples
