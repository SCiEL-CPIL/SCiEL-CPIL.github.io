"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    
    deadline = time.time() + 110  # 10s safety margin before 120s kill
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def greedy_add(S, candidates):
        """
        Greedily add elements from candidates to S while maintaining sum-free property.
        Returns the new set.
        """
        S_set = set(S)
        # Precompute which pairs in S_sum to S_set
        # For each candidate c, we need to check:
        # 1. For all a in S_set: a + c not in S_set and c + a not in S_set (same thing)
        # 2. For all a in S_set: a + c != c (i.e., a != 0, but 0 not in S)
        # 3. c + c not in S_set
        # 4. For all a in S_set: a + c not in S_set (already covered)
        # Actually, we need: for all a, b in S_set ∪ {c}, a+b not in S_set ∪ {c}
        # This is equivalent to:
        # - For all a, b in S_set: a+b not in S_set (already true if S is sum-free)
        # - For all a in S_set: a+c not in S_set and a+c != c (a!=0)
        # - c+c not in S_set and c+c != c (c!=0)
        # - For all a in S_set: c+a not in S_set (same as a+c)
        
        result = list(S)
        for c in candidates:
            if c in S_set:
                continue
            valid = True
            # Check c+c
            if 2*c in S_set or 2*c == c:  # 2c==c implies c=0, but c!=0
                valid = False
            # Check a+c for all a in S_set
            if valid:
                for a in S_set:
                    s = a + c
                    if s in S_set or s == c:  # s==c implies a=0
                        valid = False
                        break
            if valid:
                result.append(c)
                S_set.add(c)
        return result
    
    def local_search(S, candidates, max_swaps=500):
        """
        Try to improve S by swapping elements. 
        S is the current sum-free subset.
        candidates are elements not in S that could potentially be added.
        We try: remove x from S, add y from candidates, if still sum-free and size doesn't decrease.
        Actually, we want to increase size, so we try: add y, then remove minimal x to fix conflicts.
        """
        S_set = set(S)
        S_list = list(S)
        cand_set = set(candidates) - S_set
        
        # Build a conflict map: for each element y in candidates, which elements in S conflict with y?
        # y conflicts with x in S if:
        #   x + y in S_set or x + y == y (x=0) or 2*y in S_set or 2*y == y
        #   Also, for all a in S_set: a + y not in S_set (this is the main check)
        # Let's define: can_add(S_set, y) -> bool
        
        def can_add(S_set, y):
            if y in S_set:
                return False
            if 2*y in S_set:
                return False
            for a in S_set:
                if a + y in S_set:
                    return False
            return True
        
        # Try to add elements one by one, and if that fails, try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate
            for y in list(cand_set):
                if can_add(S_set, y):
                    S_set.add(y)
                    S_list.append(y)
                    cand_set.discard(y)
                    improved = True
            # If we added something, restart the loop
            if improved:
                continue
            # Try swaps: remove x from S, add y from candidates
            # We want to find y such that after removing some x, y can be added
            # For each y in candidates, find the set of x in S that block y
            for y in list(cand_set):
                # Find which x in S block y
                blockers = set()
                if 2*y in S_set:
                    blockers.add(2*y)
                for a in S_set:
                    s = a + y
                    if s in S_set:
                        blockers.add(s)
                    if a == 0:  # a+y=y, but 0 not in S
                        pass
                # Also need: for all a in S_set, a+y not in S_set
                # The blockers are the elements s = a+y that are in S_set
                # If we remove all blockers, can y be added?
                # But we can only remove one element at a time (swap)
                # So if len(blockers) == 1, we can swap
                if len(blockers) == 1:
                    x = blockers.pop()
                    # Try the swap
                    new_S_set = S_set - {x}
                    new_S_set.add(y)
                    # Verify it's sum-free
                    if is_sum_free(new_S_set):
                        S_set = new_S_set
                        S_list = list(S_set)
                        cand_set.discard(y)
                        improved = True
                        break
                elif len(blockers) > 1:
                    # Try removing each blocker and see if y can be added
                    for x in blockers:
                        new_S_set = S_set - {x}
                        if can_add(new_S_set, y):
                            new_S_set.add(y)
                            if is_sum_free(new_S_set):
                                S_set = new_S_set
                                S_list = list(S_set)
                                cand_set.discard(y)
                                improved = True
                                break
                    if improved:
                        break
        return S_list
    
    def solve_one(values, time_budget=30):
        """Find a large sum-free subset of values."""
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        # Start with odd numbers
        odd_subset = [x for x in values if x % 2 != 0]
        
        # Try greedy from different orderings
        best = odd_subset
        best_set = set(best)
        
        # Ordering 1: by absolute value (small first)
        for ordering in [
            sorted(values, key=abs),
            sorted(values, key=abs, reverse=True),
            sorted(values),
            sorted(values, reverse=True),
            values,
        ]:
            if time.time() > deadline:
                break
            # Greedy: try to add elements not in odd_subset
            candidates = [x for x in ordering if x not in best_set]
            trial = greedy_add(list(best), candidates)
            if len(trial) > len(best):
                best = trial
                best_set = set(best)
        
        # Local search
        if time.time() < deadline:
            candidates = [x for x in values if x not in best_set]
            improved = local_search(best, candidates)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Try multiple random restarts with greedy
        rng = Random(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            # Random ordering
            shuffled = list(values)
            rng.shuffle(shuffled)
            trial = greedy_add([], shuffled)
            if len(trial) > len(best):
                best = trial
                best_set = set(best)
                # Local search on the new best
                candidates = [x for x in values if x not in best_set]
                improved = local_search(best, candidates)
                if len(improved) > len(best):
                    best = improved
                    best_set = set(best)
        
        return best
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        values = instance(size, seed)
        subset = solve_one(values)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
