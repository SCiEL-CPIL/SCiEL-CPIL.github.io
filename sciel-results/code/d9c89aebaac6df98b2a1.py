"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def greedy_build(values, random_order=False, seed=0):
        """Greedy: add elements if they don't create a sum-free violation."""
        if random_order:
            rng = Random(seed)
            order = list(values)
            rng.shuffle(order)
        else:
            order = list(values)

        S = set()
        for x in order:
            # Check if adding x violates sum-free property
            # Violation if: exists a in S such that a + x in S (x = a + b -> a+b=x)
            #               or exists b in S such that x + b in S (x = a + b -> x+b=c? no)
            # Actually: a+b=c where a,b,c all in S∪{x}
            # New violations involving x:
            # 1) a + x = c: a in S, c in S
            # 2) x + b = c: b in S, c in S (same as 1 by symmetry of a,b)
            # 3) a + b = x: a in S, b in S
            valid = True
            for a in S:
                if a + x in S:
                    valid = False
                    break
                if x + a in S:
                    valid = False
                    break
                if a + a == x and a != x:  # a+a=x, but a!=x always since x not in S yet
                    valid = False
                    break
            if valid:
                S.add(x)
        return S

    def local_search(S, values, max_iter=500):
        """Try to improve S by removing elements that cause conflicts, or adding excluded ones."""
        S_set = set(S)
        excluded = set(values) - S_set
        
        for _ in range(max_iter):
            if time.time() > DEADLINE - 10:
                break
            # Try adding excluded elements
            improved = False
            for x in list(excluded):
                valid = True
                for a in S_set:
                    if a + x in S_set or x + a in S_set:
                        valid = False
                        break
                    if a + a == x:
                        valid = False
                        break
                if valid:
                    S_set.add(x)
                    excluded.discard(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one element to allow more additions
                best_removal = None
                best_gain = 0
                for a in list(S_set):
                    S_test = S_set - {a}
                    # Count how many excluded can be added
                    gain = 0
                    for x in excluded:
                        valid = True
                        for b in S_test:
                            if b + x in S_test or x + b in S_test:
                                valid = False
                                break
                            if b + b == x:
                                valid = False
                                break
                        if valid:
                            gain += 1
                    if gain > best_gain + 1:  # net gain > 0
                        best_gain = gain
                        best_removal = a
                
                if best_removal is not None:
                    S_set.discard(best_removal)
                    excluded.add(best_removal)
                    # Re-add what we can
                    for x in list(excluded):
                        valid = True
                        for a in S_set:
                            if a + x in S_set or x + a in S_set:
                                valid = False
                                break
                            if a + a == x:
                                valid = False
                                break
                        if valid:
                            S_set.add(x)
                            excluded.discard(x)
                else:
                    break
        
        return S_set

    def solve_instance(values):
        """Find a large sum-free subset of values."""
        best = set()
        
        # Strategy 1: Odd numbers
        odd = {x for x in values if x % 2 != 0}
        if len(odd) > len(best):
            best = set(odd)
        
        # Strategy 2: Positive numbers (if sum of two positives is positive, need to check)
        # Actually positives aren't automatically sum-free. Skip.
        
        # Strategy 3: Greedy in different orders
        for seed in [0, 1, 42, 123, 999]:
            if time.time() > DEADLINE - 10:
                break
            S = greedy_build(values, random_order=True, seed=seed)
            if len(S) > len(best):
                best = S
        
        # Strategy 4: Greedy sorted by absolute value
        S = greedy_build(sorted(values, key=abs), random_order=False)
        if len(S) > len(best):
            best = S
        
        # Strategy 5: Local search from best
        if time.time() < DEADLINE - 15:
            improved = local_search(best, values, max_iter=200)
            if len(improved) > len(best):
                best = improved
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE - 5:
            # Fallback: just return odd numbers
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        S = solve_instance(vals)
        results.append(sorted(S))
    
    return results
# EVOLVE-BLOCK-END
