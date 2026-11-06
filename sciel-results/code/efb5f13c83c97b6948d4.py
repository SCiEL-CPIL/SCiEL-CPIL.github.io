"""Rewritten program: Schur-graph 2-coloring + local search for sum-free subsets."""
import time
import math
from random import Random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(elements, time_budget):
        """Find a large sum-free subset of elements."""
        n = len(elements)
        elem_set = set(elements)
        idx_of = {v: i for i, v in enumerate(elements)}

        # Build conflict graph: edge (i,j) means elements[i] + elements[j] is in the set
        # Actually for sum-free: we need no a+b=c with a,b,c in subset.
        # This is a 3-uniform hypergraph constraint. But we can model it as:
        # If we pick a and b, we cannot pick a+b.
        # Alternatively: for each pair (a,b), if a+b is in set, then not all three can be in subset.
        # 
        # Better approach: model as a graph where we try to 2-color.
        # Define: for each x in elements, if 2x is in elements, then x and 2x conflict (can't both be in sum-free set).
        # Also for each x,y: if x+y is in elements, then having both x and y means x+y can't be in set.
        # This is a 3-SAT-like constraint, not simply 2-coloring.
        #
        # Let's use a different structural approach:
        # The parity baseline (all odd numbers) is always sum-free.
        # We can also use: all numbers with the same sign and same "large enough" property.
        # 
        # Actually, let's think about this more carefully.
        # Sum-free means: for any a, b in subset (a=b allowed), a+b is NOT in subset.
        # 
        # Key structural observations:
        # 1. All odd numbers are sum-free (odd+odd=even).
        # 2. All numbers in (n/2, n] for positive numbers are sum-free among themselves.
        # 3. A mix of large positives and small negatives can work.
        #
        # Strategy: Start with parity (odds), then try to add evens via local search.
        # Also try alternative initial solutions and pick the best.

        def is_sum_free(subset_set):
            """Check if a set is sum-free."""
            for a in subset_set:
                for b in subset_set:
                    if a + b in subset_set:
                        return False
            return True

        def count_conflicts(subset_set):
            """Count number of (a,b) pairs where a+b is in subset."""
            cnt = 0
            for a in subset_set:
                for b in subset_set:
                    if a + b in subset_set:
                        cnt += 1
            return cnt

        # Build adjacency: for each element x, which other elements y satisfy x+y in elements?
        # And for each element z, which pairs (x,y) satisfy x+y=z?
        
        # For local search: given current subset S, try adding an element e not in S.
        # e can be added if for all a in S, a+e not in S AND for all b in S, e+b not in S (same condition)
        # AND 2*e not in S.
        
        # Actually: e can be added to S if:
        # - For all a in S: (a + e) not in S  [covers a=e case: 2e not in S]
        # - For all b in S: (e + b) not in S  [same as above by commutativity]
        # So just: for all a in S, (a + e) not in S.

        # Precompute: for each element e, the set of elements a such that a+e is also in elements.
        # This helps check if adding e to S creates a conflict.
        
        # Let's use a smarter representation.
        # elements is sorted. n is at most 240.
        
        # Approach 1: Greedy with multiple orderings + local search
        # Approach 2: Spectral/heuristic on a derived graph
        
        # Let's try multiple strategies and keep the best.
        
        best_subset = set()
        best_size = 0

        # Strategy 1: Parity (odd numbers)
        odds = {x for x in elements if x % 2 != 0}
        if is_sum_free(odds):
            best_subset = odds
            best_size = len(odds)

        # Strategy 2: Greedy by absolute value descending (large numbers less likely to be sums)
        for order in [
            sorted(elements, key=lambda x: -abs(x)),
            sorted(elements, key=lambda x: x),
            sorted(elements, key=lambda x: -x),
            sorted(elements, key=lambda x: abs(x)),
            list(elements),
        ]:
            S = set()
            for e in order:
                # Check if e can be added
                ok = True
                for a in S:
                    if a + e in S:
                        ok = False
                        break
                if ok:
                    S.add(e)
            if len(S) > best_size:
                best_size = len(S)
                best_subset = S

        # Strategy 3: Local search from parity and from greedy solutions
        # We'll do a simple local search: try swapping one element out and another in, or just adding.
        
        def local_search(initial_set, max_iters=20000, time_limit=None):
            S = set(initial_set)
            current_size = len(S)
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                if time_limit is not None and time.time() - t0 > time_limit:
                    break
                
                # Try adding each element not in S
                candidates_to_add = [e for e in elements if e not in S]
                # Sort by number of conflicts (fewer conflicts first)
                def conflicts_if_added(e):
                    cnt = 0
                    for a in S:
                        if a + e in S:
                            cnt += 1
                    return cnt
                
                candidates_to_add.sort(key=conflicts_if_added)
                
                for e in candidates_to_add:
                    if conflicts_if_added(e) == 0:
                        S.add(e)
                        current_size += 1
                        improved = True
                        # After adding, re-sort might be expensive; just continue
                        break
                
                if improved:
                    continue
                
                # Try removing one element and adding two
                # Or: try swapping - remove e1, add e2 if it improves
                # Simpler: try to find any single addition we missed, or do a 1-1 swap
                # 1-1 swap: remove x from S, add y not in S, if result is still sum-free and size same (no gain)
                # Actually 1-1 swap doesn't increase size. We need 1-2 or 2-1.
                
                # Try: remove one element x, then see if we can add two elements y, z
                # This is expensive. Let's try a simpler perturbation.
                
                # Try random restarts of local search within the budget
                # For now, try to add elements in random order
                Random(42).shuffle(candidates_to_add)
                for e in candidates_to_add:
                    if conflicts_if_added(e) == 0:
                        S.add(e)
                        current_size += 1
                        improved = True
                        break
            
            return S

        # Apply local search to parity solution
        ls_result = local_search(best_subset, max_iters=50000)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_subset = ls_result

        # Strategy 4: More aggressive local search with random restarts
        rng = Random(12345)
        for restart in range(5):
            if time.time() - t0 > DEADLINE - 10:
                break
            # Start from a random subset of the current best
            S = set(rng.sample(list(best_subset), k=max(1, len(best_subset) // 2)))
            # Or start from scratch with random greedy
            S = set()
            random_order = elements[:]
            rng.shuffle(random_order)
            for e in random_order:
                ok = True
                for a in S:
                    if a + e in S:
                        ok = False
                        break
                if ok:
                    S.add(e)
            
            # Local search
            S = local_search(S, max_iters=20000, time_limit=DEADLINE - 5)
            if len(S) > best_size:
                best_size = len(S)
                best_subset = S

        # Strategy 5: Try to build from "large" numbers
        # Numbers with large absolute value are less likely to be expressed as a sum of two others in the set
        # (if the set is bounded)
        
        # Final verification
        if not is_sum_free(best_subset):
            # Fallback to parity
            best_subset = {x for x in elements if x % 2 != 0}
        
        return sorted(best_subset)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - t0 > DEADLINE - 5:
            # Not enough time; return parity for remaining
            elems = instance(size, seed)
            results.append(sorted([x for x in elems if x % 2 != 0]))
            continue
        elems = instance(size, seed)
        budget = min(DEADLINE - 5 - (time.time() - t0), 30.0)
        result = solve_instance(elems, budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
