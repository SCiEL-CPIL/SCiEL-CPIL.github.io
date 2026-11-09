"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if a set is sum-free: no a+b=c with a,b,c in set (a=b allowed)."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def greedy_sum_free(values, max_iter=50000):
        """
        Greedy approach: try to build a large sum-free subset.
        Strategy: 
        1. Start with all elements, then remove violations.
        2. Or: try different orderings and greedily add elements.
        3. Use local search to improve.
        """
        n = len(values)
        if n <= 2:
            return list(values)
        
        best = []
        best_size = 0
        
        # Try multiple random orderings
        for trial in range(30):
            if time.time() > DEADLINE:
                break
            
            # Random permutation
            perm = list(values)
            random.shuffle(perm)
            
            # Greedy: add element if it doesn't create a violation
            subset = set()
            for x in perm:
                # Check: for all a,b in subset, a+b != x, and x+a != b, and x+x != b
                ok = True
                for a in subset:
                    if x + a in subset:
                        ok = False
                        break
                    if x + x in subset:
                        ok = False
                        break
                if ok:
                    subset.add(x)
            
            if len(subset) > best_size:
                best_size = len(subset)
                best = list(subset)
        
        # Local search: try to swap elements to improve
        # Represent as set for fast lookup
        best_set = set(best)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try adding each excluded element and removing one conflicting element
            excluded = [x for x in values if x not in best_set]
            random.shuffle(excluded)
            
            for x in excluded[:50]:  # limit work
                if time.time() > DEADLINE:
                    break
                # Find all conflicts: pairs (a,b) in best_set such that a+b=x, or x+a=b, or x+x=b
                conflicts = set()
                for a in best_set:
                    if x + a in best_set:
                        conflicts.add(a)
                        conflicts.add(x + a)
                    if x + x in best_set:
                        conflicts.add(x + x)
                if len(conflicts) <= 1:
                    # Can add x by removing at most one element
                    if len(conflicts) == 1:
                        to_remove = list(conflicts)[0]
                        if to_remove != x:  # should be in best_set
                            best_set.remove(to_remove)
                            best_set.add(x)
                            improved = True
                    else:
                        best_set.add(x)
                        improved = True
        
        # Try to remove elements that are "bad" and see if we can add more
        # Actually, let's also try a different strategy: start with all, remove minimal set
        # to make it sum-free (this is a hitting set problem)
        
        # Alternative: try the "large positive" or "large negative" heuristic
        # Elements with large absolute value are less likely to form sums
        
        # Also try: take all elements above some threshold, or all odd, etc.
        # But we already have random greedy. Let's also try a more systematic approach.
        
        return best

    def improved_greedy(values):
        """
        Better approach: 
        1. Start with all elements.
        2. Repeatedly find a violation (a+b=c) and remove the "least useful" element.
        3. Use multiple restarts with different removal heuristics.
        """
        n = len(values)
        if n <= 2:
            return list(values)
        
        best = []
        best_size = 0
        
        for trial in range(20):
            if time.time() > DEADLINE:
                break
            
            subset = set(values)
            
            # Find violations and remove elements
            for _ in range(1000):
                if time.time() > DEADLINE:
                    break
                
                # Find a violation
                violation = None
                # Check all pairs
                items = list(subset)
                for i in range(len(items)):
                    a = items[i]
                    for j in range(i, len(items)):
                        b = items[j]
                        c = a + b
                        if c in subset:
                            violation = (a, b, c)
                            break
                    if violation:
                        break
                
                if not violation:
                    break
                
                a, b, c = violation
                # Decide which to remove: remove the one with fewest "partners"
                # or randomly choose
                # Heuristic: remove c (the sum) as it's often "derived"
                # Or remove the element that appears in most violations
                candidates = [a, b, c]
                # Count how many times each appears in violations
                # Simple: just remove c
                subset.discard(c)
                # Or try all three and pick the one that leads to fewer future violations
                # For speed, just remove c
                # Actually, let's try removing each and see which is better
                # But that's expensive. Let's use a simple heuristic.
                # Remove the element with the largest absolute value? No, that might be bad.
                # Remove randomly from the three?
                # Let's remove the one that is "most central" - appears in most sums
                # For now, just remove c
                # Actually, let's try a smarter approach: remove the element that
                # participates in the most sum relations
                # But computing that is expensive. Let's just remove c.
            
            if len(subset) > best_size:
                best_size = len(subset)
                best = list(subset)
        
        return best

    def local_search_improve(subset, values):
        """Try to improve a sum-free subset via local search."""
        subset_set = set(subset)
        values_set = set(values)
        excluded = values_set - subset_set
        
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            
            improved = False
            # Try adding an excluded element
            excl_list = list(excluded)
            random.shuffle(excl_list)
            
            for x in excl_list[:20]:
                # Check if adding x creates violations
                conflicts = set()
                for a in subset_set:
                    if x + a in subset_set:
                        conflicts.add(a)
                        conflicts.add(x + a)
                    if x + x in subset_set:
                        conflicts.add(x + x)
                
                if len(conflicts) <= 1:
                    if len(conflicts) == 1:
                        to_remove = list(conflicts)[0]
                        subset_set.discard(to_remove)
                        subset_set.add(x)
                    else:
                        subset_set.add(x)
                    excluded.discard(x)
                    if len(conflicts) == 1:
                        excluded.add(list(conflicts)[0])
                    improved = True
                    break
            
            if not improved:
                # Try removing an element and adding two?
                # Or just stop
                break
        
        return list(subset_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback to odd numbers
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        
        # Get initial solution from odd numbers
        odd_subset = [x for x in vals if x % 2]
        
        # Try greedy approaches
        greedy_result = greedy_sum_free(vals)
        
        # Try improved greedy
        improved_result = improved_greedy(vals)
        
        # Local search on the best so far
        candidates = [odd_subset, greedy_result, improved_result]
        best_candidate = max(candidates, key=len)
        
        # Local search
        if time.time() < DEADLINE:
            best_candidate = local_search_improve(best_candidate, vals)
        
        # Verify and add
        if is_sum_free(best_candidate):
            results.append(best_candidate)
        else:
            # Fallback
            results.append(odd_subset)
    
    return results
# EVOLVE-BLOCK-END
