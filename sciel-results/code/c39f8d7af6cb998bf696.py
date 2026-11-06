"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_one(vals):
        if not vals:
            return []
        
        # Precompute potential conflicts
        # We want to find a subset S such that no a,b,c in S satisfy a+b=c.
        # This is equivalent to finding an independent set in a graph where
        # edges exist between pairs that "conflict" with a third element in the set.
        # Actually, it's a 3-uniform hypergraph independent set.
        
        # Greedy heuristic:
        # Maintain a set of 'active' candidates.
        # At each step, pick the element x that minimizes the number of 
        # "violations" it would cause if added to the current solution.
        # A violation is a triple (a, b, x) or (a, x, b) or (x, a, b) 
        # where a, b are already in the solution or are x itself.
        
        # To make this efficient, we can precompute for each x, which other elements y 
        # are such that x+y is in the set, or y-x is in the set, etc.
        
        # Let's use a simpler greedy: 
        # Sort elements. Try to add each. If it causes a conflict, skip? 
        # No, that's not optimal.
        
        # Better Greedy:
        # 1. Start with all elements as candidates.
        # 2. Repeatedly pick the element with the smallest "degree" of conflict 
        #    with the current chosen set.
        #    Actually, standard greedy for independent set in graphs is: 
        #    pick vertex with min degree, add to independent set, remove it and its neighbors.
        #    Here "neighbors" are complex.
        
        # Let's try a different greedy:
        # 1. Start with empty set S.
        # 2. For each element x in vals (sorted by some heuristic, e.g., magnitude),
        #    check if adding x to S creates a sum-free violation.
        #    If not, add it.
        #    If yes, maybe skip it, or replace something?
        
        # Let's try: Sort by absolute value? Or just iterate.
        # A known heuristic for sum-free sets is to take the largest elements.
        # Why? If A is sum-free, then 2A is disjoint from A. 
        # If we take elements from the top half of the range, sums are likely out of range or not in the set.
        # For random sets in [-10000, 10000], taking elements > 5000 ensures a+b > 10000, so no sum is in the set (since max is 10000).
        # Wait, the set contains elements in [-10000, 10000]. 
        # If I pick x, y > 5000, then x+y > 10000, so x+y cannot be in the set.
        # So any subset of {x in vals : x > 5000} is sum-free.
        # Similarly, any subset of {x in vals : x < -5000} is sum-free (sums < -10000).
        # What about mixed signs?
        
        # Let's count how many elements are in (5000, 10000] and [-10000, -5000).
        # For n=60, 120, 240, roughly half should be positive, half negative.
        # So ~1/4 are > 5000, ~1/4 are < -5000. Total ~1/2 of the set is "safe" by magnitude.
        # The current "odd" strategy gets ~1/2 of the set.
        # Can we do better?
        
        # Strategy: 
        # 1. Take all x > 5000.
        # 2. Take all x < -5000.
        # 3. For the remaining "middle" elements, try to add them if they don't conflict 
        #    with the already selected large/small elements or each other.
        
        # Let's implement this specific structural insight + local improvement.
        
        pos_large = [x for x in vals if x > 5000]
        neg_large = [x for x in vals if x < -5000]
        
        # These two sets are internally sum-free and mutually sum-free 
        # (sum of pos and neg could be in range, but let's check).
        # If a > 5000 and b < -5000, a+b could be anything.
        # Example: 6000 + (-5001) = 999. 999 might be in the set.
        # So we can't just take both blindly if we want to be safe without checking.
        # However, pos_large alone is sum-free. neg_large alone is sum-free.
        # Union might not be.
        
        # Let's try to maximize size.
        # Candidate set S = pos_large + neg_large.
        # Check conflicts within S.
        # If there are conflicts, we need to remove some.
        # Alternatively, maybe just take pos_large? Size ~ n/4.
        # Current odd strategy size ~ n/2.
        # So pos_large is worse.
        
        # Let's look at the "Odd" strategy again.
        # Odds are sum-free. Size ~ n/2.
        # Can we add any evens?
        # An even number 2k conflicts with k+k=2k (so k cannot be in set if 2k is).
        # And conflicts with a+b=2k.
        
        # Let's try a local search starting from the Odd set.
        # Start with S = {x in vals : x % 2 != 0}.
        # Try to add evens.
        # For each even e in vals \ S:
        #   Check if e can be added to S without violation.
        #   Violation if:
        #     1. e/2 in S (since e/2 + e/2 = e)
        #     2. There exist a, b in S such that a + b = e.
        #   If no violation, add e to S.
        
        # Also, try to swap?
        # If adding e requires removing k (because k+k=e), is it worth it?
        # We gain 1 (e), lose 1 (k). No net gain.
        # But if adding e requires removing k and m (k+m=e), we gain 1, lose 2. Bad.
        # So only add e if no elements in S sum to e and e/2 not in S.
        
        # Let's implement this greedy addition of evens to the odd base.
        
        S = set(x for x in vals if x % 2 != 0)
        
        # Precompute sums of pairs in S? Too slow if S is large.
        # But n is small (60, 120, 240).
        # S size ~ 30, 60, 120.
        # Pairs ~ 1800, 3600, 14400.
        # We can compute all sums of pairs in S and store them in a set `sums_S`.
        # Also store S itself for O(1) lookup.
        
        sums_S = set()
        # Optimization: only compute sums a+b where a <= b to avoid duplicates? 
        # We just need to know if e is in the set of sums.
        # Iterating over all pairs is O(|S|^2).
        # For |S|=120, 14400 ops. Fast.
        
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums_S.add(s_list[i] + s_list[j])
        
        # Now try to add evens
        evens = [x for x in vals if x % 2 == 0]
        # Sort evens? Maybe by "safety"? 
        # An even e is safer if e/2 is not in S and e is not in sums_S.
        # If we add e, we must remove any k in S such that k+k=e (i.e., k=e/2).
        # And any pair (a,b) in S such that a+b=e.
        # If we just add e without removing, it's invalid.
        
        # Let's try a simple greedy:
        # Iterate through evens.
        # For each even e:
        #   Check if e is "compatible" with current S.
        #   Compatible means:
        #     - e/2 not in S (so e/2+e/2 != e violation)
        #     - No a, b in S such that a+b=e.
        #   If compatible, add e to S.
        #   Update sums_S? 
        #     If we add e, new sums are e+s for all s in S.
        #     We need to update sums_S to include these new sums for future checks.
        
        # This is a greedy independent set heuristic.
        
        new_S = set(S)
        new_sums = set(sums_S)
        
        # It might be beneficial to process evens in a specific order.
        # Perhaps smaller absolute value first? Or random?
        # Let's try sorting by absolute value ascending.
        
        for e in sorted(evens, key=abs):
            # Check conflict 1: e/2 in new_S?
            if e % 2 == 0:
                half = e // 2
                if half in new_S:
                    continue
            
            # Check conflict 2: any a, b in new_S such that a+b=e?
            # We have new_sums which contains sums of pairs from new_S.
            # Note: new_sums was built from initial S. 
            # As we add elements, we must update new_sums.
            
            if e in new_sums:
                continue
            
            # If we get here, e is compatible.
            # Add e to new_S.
            new_S.add(e)
            
            # Update new_sums with sums involving e.
            # Sums are e + s for all s in new_S (including e itself, so 2e).
            # We need to add these to new_sums.
            # Note: new_S now contains e.
            # We should iterate over the OLD new_S before adding e? 
            # No, 2e is a valid sum (e+e).
            # So we iterate over current new_S (which includes e).
            # Wait, if we iterate over new_S including e, we add e+e=2e.
            # And e+s for s in old S.
            # This is correct.
            
            # To avoid modifying set while iterating, make a list or iterate over copy.
            # But we are adding to new_sums, not new_S.
            # So we can iterate over new_S.
            for s in list(new_S):
                new_sums.add(e + s)
                
        return list(new_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Return whatever we have, padded with empty lists if needed
            while len(results) < 3:
                results.append([])
            break
        
        vals = instance(size, seed)
        res = solve_one(vals)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
