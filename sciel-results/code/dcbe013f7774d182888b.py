def construct():
    import time
    import random
    from collections import defaultdict

    deadline = time.time() + 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        start_time = time.time()
        vals_set = set(vals)
        
        # Build conflict graph for MIS formulation
        # Two elements a, b conflict if they cannot both be in a sum-free set
        # a and b conflict if:
        #   - a + b is in vals (then if both a,b in S, a+b must not be in S, but we want to exclude conflicts)
        #   Actually, the constraint is: for all a,b,c in S, a+b != c
        # This is a 3-uniform hypergraph, not pairwise.
        # 
        # Alternative: Use the fact that S is sum-free iff S ∩ (S+S) = ∅
        # We can use a different approach: greedy with backtracking or local search
        # with better conflict detection.
        
        # Let's use a smarter approach: 
        # 1. Start with structural candidates (odds, positives, negatives, large magnitude)
        # 2. Use iterative improvement with proper sum-free checking
        # 3. Add random restarts with different orderings
        
        def can_add(S, x):
            """Check if x can be added to S while maintaining sum-free."""
            if 2*x in S:
                return False
            for y in S:
                if x + y in S:
                    return False
                if x - y in S:
                    return False
            return True

        def greedy_order(ordering):
            S = set()
            for x in ordering:
                if can_add(S, x):
                    S.add(x)
            return S

        def local_search(S, max_iter=1000):
            """Improve S by trying swaps and additions."""
            S = set(S)
            local_rng = random.Random(hash(frozenset(S)))
            best_S = set(S)
            best_size = len(S)
            stuck_count = 0
            
            for _ in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                improved = False
                
                # Try to add a random element
                candidates_to_add = [x for x in vals if x not in S and can_add(S, x)]
                if candidates_to_add:
                    x = local_rng.choice(candidates_to_add)
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        stuck_count = 0
                    else:
                        S.discard(x)
                        stuck_count += 1
                else:
                    stuck_count += 1
                
                # Try a swap: remove y, add x
                if not improved and S:
                    y = local_rng.choice(list(S))
                    S.remove(y)
                    added_any = False
                    for x in vals:
                        if x in S:
                            continue
                        if can_add(S, x):
                            S.add(x)
                            added_any = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                                stuck_count = 0
                            break
                    if not added_any:
                        S.add(y)
                        stuck_count += 1
                    elif len(S) <= best_size:
                        # Revert
                        for x in list(S):
                            if x != y and x not in best_S:
                                S.discard(x)
                        S.add(y)
                        stuck_count += 1
                
                if stuck_count > 50:
                    break
            
            return best_S

        # Generate diverse starting points
        candidates = []
        
        # Structural candidates
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        sorted_vals = sorted(vals, reverse=True)
        candidates.append(set(sorted_vals[:n//2 + 1]))
        candidates.append(set(sorted_vals[-(n//2 + 1):]))
        
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        candidates.append(set(sorted_abs[:n//2 + 1]))
        candidates.append(set(sorted_abs[-(n//2 + 1):]))
        
        # Random samples with different seeds
        for seed_val in [42, 123, 456, 789, 101, 202, 303, 404]:
            rng = random.Random(seed_val)
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
        
        # Greedy with different orderings
        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
            sorted(vals, key=lambda x: random.Random(hash(x)).random()),
            sorted(vals, key=lambda x: random.Random(hash(x)+1).random()),
            sorted(vals, key=lambda x: random.Random(hash(x)+2).random()),
            sorted(vals, key=lambda x: x % 7),
            sorted(vals, key=lambda x: (x % 11)),
        ]
        
        for ordering in orderings:
            S = greedy_order(ordering)
            candidates.append(S)
        
        # Sort by size, largest first
        candidates.sort(key=len, reverse=True)
        
        best_S = set()
        best_size = -1
        
        for S in candidates:
            if time.time() - start_time > time_limit:
                break
            
            if not is_sum_free(S):
                continue
            
            S_improved = local_search(S)
            
            if not is_sum_free(S_improved):
                continue
                
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 70.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
