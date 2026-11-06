"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to S_set to keep it sum-free."""
        # x + y = z for y, z in S_set
        # y + z = x for y, z in S_set
        # x + x = z for z in S_set
        # y + y = x for y in S_set
        
        # 1. x + y = z
        # For each y in S_set, check if y + x in S_set
        # This is O(N)
        for y in S_set:
            if (y + x) in S_set:
                return False
        # 2. y + z = x
        # For each y in S_set, check if x - y in S_set
        for y in S_set:
            if (x - y) in S_set:
                return False
        # 3. x + x = z
        if (2 * x) in S_set:
            return False
        # 4. y + y = x
        # For each y in S_set, check if 2*y == x
        for y in S_set:
            if 2 * y == x:
                return False
        return True

    def greedy_construct(elements, order):
        """Construct a sum-free set by adding elements in the given order if possible."""
        S = []
        S_set = set()
        for x in order:
            if can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, elements, max_time=10):
        """Improve the set using local search."""
        S = list(S)
        S_set = set(S)
        all_elements_set = set(elements)
        excluded = all_elements_set - S_set
        rng = random.Random(123)
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            improved = False
            
            # Try to add any excluded element
            # Sort excluded by some heuristic? Or random?
            # Random is better for escaping local optima
            # But we want to add as many as possible
            # Let's try to add all possible ones repeatedly
            added_any = False
            for x in list(excluded):
                if can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    excluded.remove(x)
                    added_any = True
                    improved = True
            
            if not added_any:
                # Try swaps: remove one from S, add one from Excluded
                # This is expensive. Let's try a few random swaps
                if S and excluded:
                    # Try a few random swaps
                    for _ in range(10):
                        # Pick random x from excluded
                        x = rng.choice(list(excluded))
                        # Find conflicts of x with S
                        # If we remove y from S, does x become addable?
                        # We need to check if (S - {y}) U {x} is sum-free.
                        # This is equivalent to checking if x is addable to S - {y}
                        # And if S - {y} is still sum-free (it is, since S is)
                        
                        # We can check can_add(x, S_set - {y})
                        # But we need to iterate y.
                        # Which y to try? 
                        # Maybe y that conflicts with x?
                        # Conflicts are y such that:
                        # 1. y + x in S
                        # 2. x - y in S
                        # 3. 2*x in S (independent of y, so removing y doesn't help unless y=2x)
                        # 4. 2*y == x (independent of y? No, y is the one being removed)
                        
                        # Let's just try removing a random y and see if x fits
                        # This is O(N) check per y.
                        # We can limit the number of y's we try.
                        
                        # Optimization: Only try y's that are "close" to x or involved in conflicts
                        # But for simplicity, let's try a few random y's
                        candidates_y = rng.sample(S, min(len(S), 10))
                        for y in candidates_y:
                            temp_S_set = S_set - {y}
                            if can_add(x, temp_S_set):
                                # Perform swap
                                S.remove(y)
                                S_set.remove(y)
                                S.append(x)
                                S_set.add(x)
                                excluded.discard(x)
                                excluded.add(y)
                                improved = True
                                break
                        if improved:
                            break
                            
            if not improved:
                # Stuck. Try a random restart or perturbation?
                # Perturbation: Remove k random elements, then greedily add back
                if S:
                    k = min(len(S), 3)
                    to_remove = rng.sample(S, k)
                    for y in to_remove:
                        S.remove(y)
                        S_set.remove(y)
                        excluded.add(y)
                    # Now greedily add back
                    for x in list(excluded):
                        if can_add(x, S_set):
                            S.append(x)
                            S_set.add(x)
                            excluded.remove(x)
                    
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        
        # Generate multiple candidates
        candidates = []
        
        # 1. Greedy Descending
        candidates.append(greedy_construct(elements, sorted(elements, reverse=True)))
        
        # 2. Greedy Ascending
        candidates.append(greedy_construct(elements, sorted(elements)))
        
        # 3. Greedy Random (multiple runs)
        rng = random.Random(seed)
        for _ in range(5):
            shuffled = elements[:]
            rng.shuffle(shuffled)
            candidates.append(greedy_construct(elements, shuffled))
        
        # 4. Top Half (heuristic)
        # Take elements > 0?
        pos = [x for x in elements if x > 0]
        candidates.append(greedy_construct(elements, sorted(pos, reverse=True)))
        
        # Select best candidate
        best_candidate = max(candidates, key=len)
        
        # Local Search
        # Allocate time based on instance size
        time_limit = 10 if size > 100 else 5
        improved_set = local_search(best_candidate, elements, max_time=time_limit)
        
        # Verify
        if not is_sum_free(improved_set):
            improved_set = best_candidate
            
        results.append(improved_set)
    
    return results
# EVOLVE-BLOCK-END
