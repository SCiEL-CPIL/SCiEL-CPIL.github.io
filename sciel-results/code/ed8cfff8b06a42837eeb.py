"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def construct():
    deadline = time.time() + 145.0  # Safety margin from 150s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(S, seed):
        n = len(S)
        if n == 0:
            return []
        
        # Build conflict graph
        # Edge (i, j) exists if S[i] + S[j] == S[k] for some k, or S[i]+S[i]==S[j], etc.
        # Actually, the condition is: no a, b, c in S such that a+b=c.
        # This means if we select a set T, for any a, b in T, a+b must not be in T.
        # We can model this as a graph where vertices are elements of S.
        # An edge (u, v) exists if u+v is in S. If we pick u and v, we cannot pick u+v.
        # This is not a simple independent set on a graph where edge means "cannot be together".
        # Wait. If u+v = w, then {u, v, w} is a forbidden triple.
        # This is a 3-uniform hypergraph independent set problem (or 3-SAT like).
        # However, we can simplify.
        # If we fix the set of "large" numbers, the "small" numbers are constrained.
        
        # Alternative approach: Greedy with bitset optimization + Local Search
        # Let's stick to a fast heuristic that is better than the naive O(n^3) check.
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # Precompute for each element x, the set of elements y such that x+y is in S.
        # And for each element z, the set of pairs (a,b) such that a+b=z.
        
        # Let's use a different strategy: Randomized Greedy + Local Search
        # The naive local search was too slow.
        
        # Strategy:
        # 1. Try multiple random orderings.
        # 2. Greedily add elements if they don't violate the sum-free property.
        # 3. Use a fast check: maintain a set of selected elements.
        #    To check if x can be added:
        #    - For all a in selected: a+x not in selected
        #    - For all a, b in selected: a+b != x
        #    This is O(n^2) per check, O(n^3) total. Still slow for n=240.
        
        # Better Strategy: Bitset-based Greedy
        # Represent selected set as a bitset.
        # Precompute for each x, a bitset of elements y such that x+y is in S.
        # Let B[x] be the bitset of indices j such that S[i] + S[j] = S[k] for some k? No.
        # Let's define conflict bitsets.
        # If we add x, we conflict with y if x+y is in S.
        # So if x is selected, y cannot be selected IF x+y is in S.
        # This is a directed conflict. If x+y=z, then (x,y) is a conflict pair relative to z.
        # Actually, if we select x and y, we must NOT select z=x+y.
        
        # This is equivalent to: Find a maximum subset T such that T is sum-free.
        # Let's try a different angle: 
        # The set of odd numbers is always sum-free (odd+odd=even).
        # The set of numbers > n/2 is sum-free (sum of two > n).
        
        # Let's implement a fast local search on the "Odd" base.
        # Start with Odds.
        # Try to swap an odd out and an even in? No, evens can be sum-free too.
        
        # Let's use a simulated annealing approach on the set of indices.
        # State: bitset of selected elements.
        # Energy: number of violations. We want 0 violations.
        # But we also want to maximize size.
        
        # Let's try a simpler, faster heuristic:
        # 1. Generate several random subsets of size ~ n/2.
        # 2. Filter to be sum-free.
        # 3. Try to expand.
        
        # Actually, the most robust method for this size (n<=240) within 150s is:
        # Greedy with bitsets for fast conflict checking.
        
        # Precompute:
        # For each i, let conflict[i] be a bitset of j such that S[i] + S[j] is in S.
        # If we select i and j, and S[i]+S[j] is in S (say at index k), then we cannot select k.
        # This is not a pairwise conflict between i and j directly, but a constraint on k.
        
        # Let's reformulate:
        # A set T is sum-free iff for all a,b in T, a+b not in T.
        # This is equivalent to: T contains no solutions to a+b=c.
        
        # We can use a branch and bound on a small core? No, 240 is too big for exhaustive.
        
        # Let's optimize the local search from the seed.
        # The seed's local search was O(n^3) per step.
        # We can make it O(n) or O(n log n) using bitsets or sets.
        
        def is_sum_free(S):
            s = set(S)
            for a in S:
                for b in S:
                    if a + b in s:
                        return False
            return True

        def greedy_add(S, rng, max_iter=50000):
            """Fast greedy addition."""
            current = set()
            # Sort by magnitude or random?
            # Randomized greedy
            candidates = list(S)
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return list(current)
                # Check if x can be added
                # 1. x+x not in current
                # 2. For all a in current, a+x not in current
                # 3. For all a, b in current, a+b != x
                
                # Optimization: 
                # If we maintain current as a set, check 1 and 2 are O(n).
                # Check 3 is O(n^2).
                
                # We can precompute sums?
                
                ok = True
                if x + x in current:
                    ok = False
                if ok:
                    for a in current:
                        if a + x in current:
                            ok = False
                            break
                if ok:
                    # Check if x is sum of two elements in current
                    # a + b = x => b = x - a
                    for a in current:
                        if x - a in current:
                            # Ensure a != x-a? No, a=b is allowed in sum-free definition?
                            # "no a + b = c with a, b, c in it (a = b allowed)"
                            # So if a = x/2 and a in current, then a+a=x, violation.
                            # The loop above checks if x-a is in current.
                            # If a = x-a, i.e., 2a=x, then we found a in current and x-a (which is a) in current.
                            # So this check covers a=b case too.
                            ok = False
                            break
                
                if ok:
                    current.add(x)
            
            return list(current)

        def local_search(S, rng, max_iters=10000):
            """
            Local search: 
            Start with a sum-free set.
            Try to remove an element and add another to increase size.
            Or swap.
            """
            # Start with odds
            current = set(x for x in S if x % 2 != 0)
            
            # Greedy add
            current = set(greedy_add(list(current), rng))
            
            best = list(current)
            best_size = len(best)
            
            # Local search loop
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                
                improved = False
                # Try to remove a random element
                if not current:
                    break
                removed = rng.choice(list(current))
                current.discard(removed)
                
                # Try to add an element from S \ current
                # Pick a random candidate? Or iterate?
                # Let's try all candidates that were previously blocked by 'removed'?
                # Or just random samples.
                
                candidates = [x for x in S if x not in current]
                if not candidates:
                    current.add(removed)
                    break
                
                # Try to add a random candidate
                # To be efficient, we can try a few
                for _ in range(10):
                    x = rng.choice(candidates)
                    # Check if x can be added
                    ok = True
                    if x + x in current:
                        ok = False
                    if ok:
                        for a in current:
                            if a + x in current:
                                ok = False
                                break
                    if ok:
                        for a in current:
                            if x - a in current:
                                ok = False
                                break
                    
                    if ok:
                        current.add(x)
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(best)
                        improved = True
                        break
                
                if not improved:
                    current.add(removed)
            
            return best

        # Run multiple restarts
        best_overall = []
        best_size = 0
        
        # Allocate time for multiple restarts
        start_time = time.time()
        restarts = 0
        while time.time() < deadline - 5:
            restarts += 1
            rng = Random(seed + restarts * 100)
            # Strategy 1: Odds + Greedy + Local Search
            res1 = local_search(S, rng)
            
            # Strategy 2: Random start + Greedy + Local Search
            # Start with random half
            rng2 = Random(seed + restarts * 200)
            random_start = set(rng2.sample(S, len(S)//2))
            # Clean random start to be sum-free
            # This is tricky, just use greedy from empty on random order
            res2 = greedy_add(S, rng2)
            
            # Strategy 3: High numbers only
            # Sort S, take top half?
            sorted_S = sorted(S)
            high_start = set(sorted_S[len(sorted_S)//2:])
            # Check if sum-free? High numbers sum to > max(S), so likely sum-free.
            # But if S has negative numbers, "high" doesn't mean sum-free.
            # However, if we take numbers > 0 and large, their sum is > max.
            # Let's just use greedy on high numbers.
            res3 = greedy_add(list(high_start), Random(seed + restarts * 300))
            
            for res in (res1, res2, res3):
                if len(res) > best_size:
                    best_size = len(res)
                    best_overall = res
            
            # If we found a very large set, we might stop early?
            # Theoretical max is n. But sum-free max is usually ~ n/2 or less.
            # If best_size > n * 0.6, it's good.
            
            # Refine best_overall with local search
            if restarts % 5 == 0:
                rng_refine = Random(seed + restarts * 400)
                refined = local_search(best_overall, rng_refine)
                if len(refined) > best_size:
                    best_size = len(refined)
                    best_overall = refined
        
        return best_overall

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        # Ensure we have time for each
        # Allocate proportional time? Or just run until deadline?
        # The loop inside solve_one handles the deadline.
        res = solve_one(S, seed)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
