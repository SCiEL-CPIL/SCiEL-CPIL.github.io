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
        result = list(S)
        for c in candidates:
            if c in S_set:
                continue
            valid = True
            # Check c+c
            if 2*c in S_set:
                valid = False
            # Check a+c and c-a for all a in S_set
            if valid:
                for a in S_set:
                    if a + c in S_set:
                        valid = False
                        break
                    if c - a in S_set:
                        valid = False
                        break
            if valid:
                result.append(c)
                S_set.add(c)
        return result
    
    def local_search(S, candidates, time_limit=5):
        """
        Try to improve S by adding or swapping elements.
        """
        S_set = set(S)
        cand_set = set(candidates) - S_set
        start_time = time.time()
        
        def can_add(S_set, y):
            if y in S_set:
                return False
            if 2*y in S_set:
                return False
            for a in S_set:
                if a + y in S_set:
                    return False
                if y - a in S_set:
                    return False
            return True
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # 1. Try additions
            for y in list(cand_set):
                if can_add(S_set, y):
                    S_set.add(y)
                    cand_set.discard(y)
                    improved = True
            if improved:
                continue
            
            # 2. Try swaps
            S_list = list(S_set)
            for y in list(cand_set):
                # Only try swaps if y cannot be added directly
                if can_add(S_set, y):
                    continue
                for x in S_list:
                    if x == y:
                        continue
                    new_S = S_set - {x}
                    if can_add(new_S, y):
                        new_S.add(y)
                        S_set = new_S
                        cand_set.discard(y)
                        improved = True
                        break
                if improved:
                    break
        return list(S_set)
    
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
        
        # Final verification and fallback
        if not is_sum_free(subset):
            # Fallback to odd numbers, which are always sum-free
            subset = [x for x in values if x % 2 != 0]
            
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
