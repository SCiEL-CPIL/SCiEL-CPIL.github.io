"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _is_coprime_conflict(a, b):
    """Check if gcd(a,b) == 1 (i.e., they conflict)."""
    return math.gcd(a, b) == 1

def construct():
    deadline = time.time() + 80  # 80s safety margin (90s limit)
    
    # Candidate pool: all numbers in [1,255] that share a factor with 255 (3, 5, or 17)
    # Since 255 must be in the set, every other element must share a factor with 255.
    # Also, every pair in the set must share a factor.
    
    # First, build the candidate set: numbers that share at least one prime factor with 255
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # 255 must be included. Index of 255 in candidates:
    idx_255 = candidates.index(255)
    
    n_cand = len(candidates)
    
    # Build conflict matrix: conflict[i][j] = True if gcd(candidates[i], candidates[j]) == 1
    # We want to find the largest subset with NO conflicts (a clique in the non-coprime graph).
    # This is equivalent to finding the largest independent set in the conflict graph,
    # or equivalently, the maximum clique in the non-coprime graph.
    
    # For MILP formulation:
    # x_i = 1 if candidate i is selected
    # Maximize sum(x_i)
    # Subject to:
    #   x_{255} = 1 (must include 255)
    #   x_i + x_j <= 1 for all pairs (i,j) where gcd(candidates[i], candidates[j]) == 1
    #   x_i in {0, 1}
    
    # Build list of conflict pairs
    conflict_pairs = []
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if _is_coprime_conflict(candidates[i], candidates[j]):
                conflict_pairs.append((i, j))
    
    # Try MILP with scipy
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = -np.ones(n_cand)
        
        # Constraints: x_i + x_j <= 1 for each conflict pair
        # A_ub @ x <= b_ub
        if conflict_pairs:
            A = np.zeros((len(conflict_pairs), n_cand))
            for k, (i, j) in enumerate(conflict_pairs):
                A[k, i] = 1
                A[k, j] = 1
            b_ub = np.ones(len(conflict_pairs))
            
            # x_255 = 1: both x_255 <= 1 and x_255 >= 1
            # We'll handle equality via bounds
            cons = LinearConstraint(A, ub=b_ub)
        else:
            cons = LinearConstraint(np.zeros((0, n_cand)), ub=np.zeros(0))
        
        # Bounds: 0 <= x_i <= 1 for all, with x_255 = 1
        lb = np.zeros(n_cand)
        ub = np.ones(n_cand)
        lb[idx_255] = 1  # Force x_255 = 1
        bounds = Bounds(lb, ub)
        
        # Integrality: all variables are integer
        integrality = np.ones(n_cand)
        
        # Check time
        if time.time() > deadline - 5:
            raise TimeoutError("Not enough time for MILP")
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={'time_limit': 70})
        
        if res.status == 0 and res.x is not None:
            selected = [candidates[i] for i in range(n_cand) if res.x[i] > 0.5]
            if 255 in selected and len(selected) > 0:
                # Verify the solution
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if math.gcd(selected[i], selected[j]) == 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return sorted(selected)
        
        # If MILP didn't find a good solution, fall through to heuristic
    except Exception:
        pass
    
    # Fallback: greedy approach - start with all multiples of 3 (size 85),
    # then try to add elements that don't conflict
    # Actually, let's try a smarter approach: use the fact that we need a clique
    # in the non-coprime graph.
    
    # Strategy: Try different "anchor" prime sets.
    # The multiples of 3 give 85. Can we do better?
    # 
    # Let's try: include all multiples of 3, then see if we can swap some out
    # for numbers that share factors with 255 but not with 3, while maintaining
    # the pairwise property.
    #
    # Actually, let's try a direct greedy: 
    # 1. Start with 255.
    # 2. Greedily add the number that conflicts with the fewest already-selected numbers.
    # But this is O(n^3) which for n=170 candidates is fine.
    
    # Better: Use the structure. Numbers sharing a factor with 255 are those div by 3, 5, or 17.
    # Let's categorize:
    div3 = set(x for x in range(1, 256) if x % 3 == 0)
    div5 = set(x for x in range(1, 256) if x % 5 == 0)
    div17 = set(x for x in range(1, 256) if x % 17 == 0)
    
    # The multiples of 3 form a clique of size 85.
    # Can we add any number from div5 \ div3 or div17 \ div3?
    # A number d in div5 \ div3 must share a factor with every element in our set.
    # If our set is all multiples of 3, then d must share a factor with 3*k for all k.
    # In particular, d must share a factor with 3*7=21, 3*11=33, etc.
    # Since d is not div by 3, d must share a factor with 7, 11, 13, 17, 19, ...
    # i.e., all primes not equal to 3 up to 85. That's impossible for a single number <= 255.
    # So we cannot add any div5\not div3 to the full set of multiples of 3.
    #
    # What if we remove some multiples of 3 to make room?
    # This becomes a complex optimization. Let's try a few specific strategies.
    
    # Strategy 1: All multiples of 3 (baseline, size 85)
    result1 = sorted(div3)
    
    # Strategy 2: Try to find a larger clique using a different approach.
    # Let's try: all numbers that are div by 3 OR div by 5, but we need pairwise gcd > 1.
    # This won't work directly because 3*7=21 and 5*11=55 are coprime.
    
    # Let's try a local search / improvement from the 85-element set.
    # Try swapping: remove a multiple of 3, add a non-multiple-of-3 that shares factors
    # with all remaining elements.
    
    # For efficiency, let's just verify that 85 is likely optimal and return it.
    # But first, let's try one more thing: what about numbers that are multiples of 15 (div by both 3 and 5)?
    # They're already in div3.
    
    # Let's try a smarter MILP with fewer variables by exploiting structure.
    # Actually, let's just return the multiples of 3 if we couldn't do better.
    
    if time.time() > deadline - 2:
        return result1
    
    # Try a simple local search: for each element in result1, try removing it and
    # adding up to 2 elements from candidates \ result1 that don't conflict with
    # the remaining set.
    # This is expensive, so let's limit it.
    
    best = result1
    
    # Try adding elements one at a time by removing conflicting elements
    # For each candidate not in result1, check how many elements in result1 it conflicts with
    for cand in candidates:
        if cand in div3:
            continue
        # Count conflicts with result1
        conflicts = [x for x in result1 if math.gcd(cand, x) == 1]
        if len(conflicts) == 0:
            # Can add without removing anything!
            new_set = result1 + [cand]
            if len(new_set) > len(best):
                best = new_set
        elif len(conflicts) == 1:
            # Can add by removing 1 element
            removed = conflicts[0]
            new_set = [x for x in result1 if x != removed] + [cand]
            if len(new_set) > len(best):
                best = new_set
        elif len(conflicts) <= 2:
            # Try removing up to 2 conflicting elements
            if len(conflicts) == 2:
                for i in range(2):
                    for j in range(i+1, 2):
                        pass  # Only 2 elements, just remove both
                new_set = [x for x in result1 if x not in set(conflicts)] + [cand]
                if len(new_set) > len(best):
                    best = new_set
    
    # Verify best
    if 255 not in best:
        return result1
    
    valid = True
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            if math.gcd(best[i], best[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if valid and len(best) >= len(result1):
        return sorted(best)
    
    return result1
# EVOLVE-BLOCK-END
