"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(100000)

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(S):
        """
        Find a maximum sum-free subset of S using branch-and-bound.
        """
        import time
        start_time = time.time()
        n = len(S)
        if n <= 1:
            return S[:]
        
        s_set = set(S)
        
        # Precompute all sum triples: (a, b, c) where a+b=c, all in S
        # a and b can be equal
        triples = []
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                c = a + b
                if c in s_set and c != a and c != b:
                    # Store as sorted tuple for uniqueness, but keep track of indices
                    triple = tuple(sorted([a, b, c]))
                    if triple not in [t[0] for t in triples]:
                        triples.append((triple, [a, b, c]))
        
        # Actually, let's just store the sets
        triple_sets = set()
        for a in S:
            for b in S:
                c = a + b
                if c in s_set:
                    triple_sets.add(frozenset([a, b, c]))
        
        triple_list = [list(t) for t in triple_sets]
        
        # For each element, count how many triples it's in
        elem_triple_count = {x: 0 for x in S}
        for t in triple_list:
            for x in t:
                elem_triple_count[x] += 1
        
        # Sort elements by triple count descending (high degree first)
        sorted_S = sorted(S, key=lambda x: -elem_triple_count[x])
        index_map = {x: i for i, x in enumerate(sorted_S)}
        
        # Branch and bound: start with all elements, remove to satisfy constraints
        best_size = 0
        best_set = set()
        
        # First, get a good initial solution via greedy
        greedy_set = set()
        for x in sorted_S:
            # Check if adding x creates a violation with existing elements
            valid = True
            for a in greedy_set:
                if x - a in greedy_set and x - a != a:
                    valid = False
                    break
                if x + a in greedy_set:
                    valid = False
                    break
                if x - a in greedy_set:
                    # a + (x-a) = x, so if x-a in greedy_set, violation
                    valid = False
                    break
            if valid:
                greedy_set.add(x)
        
        best_size = len(greedy_set)
        best_set = set(greedy_set)
        
        # Branch and bound
        def is_sum_free(T):
            T_set = set(T)
            for a in T_set:
                for b in T_set:
                    if a + b in T_set and a + b != a and a + b != b:
                        return False
            return True
        
        # Recursive branch and bound
        # State: current set (as frozenset for memo), index in sorted_S
        # We process elements in order, deciding to keep or remove each
        
        from functools import lru_cache
        import sys
        sys.setrecursionlimit(10000)
        
        # Use a different approach: iterative deepening or just greedy + local search
        # Given time constraints, let's do a better local search
        
        # Local search: try to improve by swapping
        current = set(greedy_set)
        
        def can_add(T, x):
            """Check if T ∪ {x} is sum-free."""
            for a in T:
                if x - a in T:
                    return False
                if x + a in T:
                    return False
            return True
        
        improved = True
        while improved and (time.time() - start_time) < 100:
            improved = False
            # Try removing each element and adding others
            for x in list(current):
                candidate = current - {x}
                # Try to add elements not in candidate
                for y in S:
                    if y not in candidate and can_add(candidate, y):
                        candidate.add(y)
                if len(candidate) > len(current):
                    current = candidate
                    improved = True
                    break
        
        # Also try adding elements to current
        for y in S:
            if y not in current and can_add(current, y):
                current.add(y)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        return list(best_set)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_sum_free(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
