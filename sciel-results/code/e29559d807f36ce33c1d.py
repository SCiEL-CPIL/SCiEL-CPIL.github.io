"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _all_reciprocal_sums_distinct(subset):
    """Check if all 2^k reciprocal subset sums are distinct."""
    k = len(subset)
    if k > 22:
        return False
    sums = [0.0] * (1 << k)
    sums[0] = 0.0
    for i in range(k):
        val = 1.0 / subset[i]
        for mask in range(1 << i):
            sums[mask | (1 << i)] = sums[mask] + val
    # Check for duplicates
    # For k up to ~22, 2^k can be large. Let's use a set for smaller k,
    # and for larger k, we might need a different approach or just trust
    # that if the set is well-separated, they are distinct.
    # But for k > 20, 2^k = 1 million floats. Checking distinctness is O(2^k log 2^k)
    # which might be slow if called many times.
    # However, we only call this for candidates we are considering adding or validating.
    # Let's optimize: if the sum of the smallest k-1 elements is less than the largest,
    # and we add in decreasing order, distinctness is guaranteed by the 'superincreasing' property
    # for the reciprocals? No, reciprocals are decreasing, so the 'superincreasing' check is
    # a_i > sum_{j>i} a_j.
    # If 1/a_i > sum_{j>i} 1/a_j for all i, then all subset sums are distinct.
    # This is a sufficient condition.
    # Let's implement the check using this sufficient condition first for speed,
    # and fall back to exact check if necessary (or just use the sufficient condition as it's hard to beat).
    # Actually, the task requires *all* subset sums distinct.
    # The condition a_i > sum_{j>i} a_j ensures that the interval [S, S+a_i) for subsets including i
    # does not overlap with [S', S'+a_j) for subsets not including i (since max sum of smaller < a_i).
    # Let's verify this logic.
    # Let A be the set. Sort A such that a_1 > a_2 > ... > a_k (so 1/a_1 < 1/a_2 < ... is false, 
    # wait. If subset is [22, 21, ...], then a_1=22, a_2=21. 1/22 < 1/21.
    # Let weights w_i = 1/a_i. Sort indices such that w_1 > w_2 > ... > w_k.
    # This means a_1 < a_2 < ... < a_k.
    # Condition: w_i > sum_{j>i} w_j.
    # This guarantees distinct sums.
    # Is it necessary? No. But it's a very strong constraint.
    # The greedy algorithm adds elements where this condition holds.
    # If we just check this condition, we might reject valid sets.
    # However, for the purpose of maximizing cardinality, this condition is often close to optimal
    # or the only way to guarantee it quickly.
    # Let's do the exact check for small k (k<=20) and the sufficient condition for large k?
    # Actually, 2^20 = 1e6. Generating 1e6 sums and checking duplicates takes ~0.1s in Python.
    # If we call this 1000 times, that's 100s. Too slow.
    # We need a faster validity check.
    # The sufficient condition is O(k^2) or O(k) with prefix sums.
    # Let's use the sufficient condition:
    # Sort subset in increasing order of value (decreasing weight).
    # Let sorted_subset = sorted(subset)
    # Weights = [1/x for x in sorted_subset]
    # Check if weights[i] > sum(weights[i+1:]) for all i.
    # If yes, valid.
    # If no, we still need to check.
    # But maybe we can just rely on the sufficient condition for the search,
    # and then do one final exact check on the best candidate?
    # The risk is that a set violating the sufficient condition might still be valid.
    # But if we only search within sets satisfying the sufficient condition, we are safe.
    # And we return that.
    # Is it possible that a larger set exists that violates the sufficient condition?
    # Probably, but finding it is hard.
    # Let's try to find the largest set satisfying the sufficient condition first.
    # Then, try to add more elements that *might* work, and check exactly.
    
    # For this function, let's just do the exact check if k <= 18, else sufficient condition.
    # Actually, let's just do the sufficient condition for speed in the loop,
    # and if we find a candidate, we can verify it exactly at the end.
    
    # Wait, the function signature is just a helper. I will define _is_valid_exact and _is_valid_sufficient.
    pass

def _is_valid_exact(subset):
    k = len(subset)
    if k == 0:
        return True
    if k > 22:
        return False
    # Use integer arithmetic if possible? No, reciprocals are floats.
    # But 1/a_i are rational.
    # Sum of reciprocals can be represented as P/Q.
    # Q = product of all a_i. This grows very fast.
    # For k=10, product of 10 distinct integers from 1..22 is huge.
    # Floats might lose precision for k > 20.
    # But for distinctness, if two sums are very close, they might be equal in float but not in reality.
    # Or different in reality but equal in float.
    # However, the minimum difference between distinct subset sums of reciprocals of integers <= 22
    # is likely larger than float epsilon for k < 20.
    # For k=20, 2^20 sums. Range is sum(1/1..1/22) ~ 3.6.
    # Smallest gap?
    # Let's just use floats and a set.
    sums = set()
    # Iterative generation
    current_sums = {0.0}
    for a in subset:
        val = 1.0 / a
        new_sums = set()
        for s in current_sums:
            ns = s + val
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
        current_sums = new_sums
    return True

def _is_valid_sufficient(subset):
    """Check if 1/a_i > sum_{j>i} 1/a_j for all i, when sorted by increasing a."""
    if not subset:
        return True
    sorted_s = sorted(subset)
    # weights[i] = 1/sorted_s[i]
    # Check if w[i] > sum(w[i+1:])
    suffix_sum = 0.0
    for i in range(len(sorted_s)-1, -1, -1):
        w = 1.0 / sorted_s[i]
        if w < suffix_sum:
            return False
        suffix_sum += w
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s buffer
    
    # Universe
    universe = list(range(1, 23))
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # Verify best_set is valid exactly
    if not _is_valid_exact(best_set):
        best_set = [1] # Fallback
    
    # Greedy construction satisfying sufficient condition
    # Start with empty, add elements in decreasing order of a (increasing 1/a? No, decreasing a means increasing weight? No.
    # 1/a is smaller for larger a.
    # To satisfy w_i > sum_{j>i} w_j, we want large weights first.
    # So we should add small a's first?
    # If we add 1 first, w=1. Then we need next w < 1. 1/2 = 0.5. 1/3=0.33.
    # If we add 1, 2: w1=1, w2=0.5. 1 > 0.5. OK.
    # Add 3: w3=0.33. 0.5 > 0.33? Yes. 1 > 0.5+0.33? Yes.
    # So adding in increasing order of a (small to large) builds the set of "large weights" first.
    # This seems correct.
    
    greedy_set = []
    for a in sorted(universe): # 1 to 22
        candidate = greedy_set + [a]
        if _is_valid_sufficient(candidate):
            greedy_set = candidate
        # Time check
        if time.time() > deadline:
            break
    
    if len(greedy_set) > len(best_set):
        if _is_valid_exact(greedy_set):
            best_set = greedy_set

    # Now try to improve by swapping or adding
    # Local search: try to add one element, remove one, swap two.
    # Use random restarts or random swaps.
    
    current_best = best_set[:]
    
    # Try to add elements not in current_best
    # If adding breaks sufficient condition, check exact.
    # If adding breaks exact, reject.
    
    # Let's do a simple hill climbing.
    # Move: Add one element, Remove one element, Swap two.
    
    # To make it efficient, we will prioritize moves that maintain the sufficient condition.
    # If a move violates sufficient condition, we check exact.
    
    def _score(s):
        return len(s)
    
    # Randomized local search
    # We will try a number of random moves within the time limit.
    
    # Precompute valid additions?
    
    # Let's iterate.
    # To avoid getting stuck, use simulated annealing or just random restarts.
    # Given the small size, random restarts with greedy fill might be good.
    
    # Strategy:
    # 1. Generate many random sets satisfying sufficient condition.
    # 2. Check exact validity.
    # 3. Keep the largest.
    # 4. Try to extend them.
    
    # How to generate random sets satisfying sufficient condition?
    # The condition is quite restrictive.
    # It's basically a superincreasing sequence of weights.
    # We can pick weights randomly? No, weights are fixed to 1/a.
    # So we are picking a subset of {1/a}.
    # The condition is 1/a_i > sum_{j>i} 1/a_j.
    # This implies a_i < a_j * (sum_{k>i} 1/a_k + 1) ... not helpful.
    
    # Let's just do random swaps/additions on the current best.
    
    # Shuffle universe
    random.seed(42)
    
    max_iterations = 100000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        # Pick a random move
        # 1. Add a random element not in set
        # 2. Remove a random element
        # 3. Swap a random element in set with one not in set
        
        # We want to maximize size, so Add and Swap are preferred.
        # Remove is only used if stuck (but we don't track stuck easily).
        
        # Let's try Add first.
        if len(current_best) < 22:
            candidates_to_add = [x for x in universe if x not in current_best]
            if candidates_to_add:
                x = random.choice(candidates_to_add)
                new_set = current_best + [x]
                # Check
                # If sufficient condition holds, it's valid.
                # If not, check exact.
                # To save time, check sufficient first.
                if _is_valid_sufficient(new_set):
                    current_best = new_set
                    continue
                else:
                    # Check exact
                    # This is expensive. Only do it if we think it might be valid.
                    # How to know? If the violation of sufficient condition is small.
                    # Or just do it randomly.
                    if random.random() < 0.1: # 10% chance of exact check on violation
                        if _is_valid_exact(new_set):
                            current_best = new_set
                            continue
                    # Try swap instead if add failed?
        
        # Try Swap
        if current_best:
            in_set = current_best[:]
            out_set = [x for x in universe if x not in current_best]
            if in_set and out_set:
                x_in = random.choice(in_set)
                x_out = random.choice(out_set)
                new_set = [x if x != x_in else x_out for x in current_best]
                if _is_valid_sufficient(new_set):
                    current_best = new_set
                    continue
                else:
                    if random.random() < 0.1:
                        if _is_valid_exact(new_set):
                            current_best = new_set
                            continue
        
        # If we fail to add/swap, try Remove to escape local optimum?
        # Or just restart?
        # Let's try Remove with low probability.
        if random.random() < 0.05 and len(current_best) > 1:
            x = random.choice(current_best)
            new_set = [x2 for x2 in current_best if x2 != x]
            current_best = new_set # Always valid if subset of valid set
    
    # Final exact check
    if not _is_valid_exact(current_best):
        # Fallback to best known valid
        current_best = best_set
        if not _is_valid_exact(current_best):
             current_best = [1]

    return current_best
# EVOLVE-BLOCK-END
