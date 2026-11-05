"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def _is_safe(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of reciprocals of any non-empty
    subset of `others`."""
    target = Fraction(1, candidate)
    n = len(others)
    if n == 0:
        return True
    
    # Meet-in-the-middle for subset sum
    mid = n // 2
    left = others[:mid]
    right = others[mid:]
    
    # Generate all subset sums for left
    left_sums = {Fraction(0, 1)}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        left_sums.update(new_sums)
    
    # Check if any left sum + right sum == target
    # For each sum in right, check if (target - right_sum) is in left_sums
    right_sums = {Fraction(0, 1)}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        right_sums.update(new_sums)
    
    # Now check combinations
    for rs in right_sums:
        if rs > target:
            continue
        remainder = target - rs
        if remainder in left_sums:
            # We need to ensure the combined subset is non-empty.
            # Since target > 0, and all terms are positive, if remainder == 0 and rs == target, 
            # that's a valid subset (non-empty since target > 0).
            # The empty set sum is 0. If rs=0 and remainder=target, it implies target in left_sums.
            # If rs=target and remainder=0, it implies target in right_sums.
            # In all cases where target > 0, a match implies a non-empty subset.
            return False
            
    return True

def _is_valid_full(subset):
    """Check if the entire subset is reciprocal-subsum-free."""
    s = set(subset)
    for x in s:
        others = list(s - {x})
        if not _is_safe(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Strategy 1: Greedy construction from largest to smallest.
    # This is valid because if we add k to S (where all elements in S are > k),
    # we only need to check if 1/k is a sum of reciprocals of S.
    # Existing elements x in S have 1/x < 1/k (since x > k).
    # Any sum involving 1/k and other reciprocals (which are < 1/k) would be > 1/k > 1/x.
    # So 1/x cannot be formed by a subset including k.
    # Thus, the set remains valid for all previous elements.
    
    selected = []
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe(i, selected):
            selected.append(i)
    
    best_set = set(selected)
    
    # Strategy 2: Try to add back excluded elements.
    # The greedy might have excluded some elements that could be added if we
    # were smarter, or perhaps the greedy order prevented adding an element
    # that is actually safe in the final context (though for decreasing order,
    # the "safe" check is sufficient for validity, so if it was rejected, it's
    # truly unsafe to add to the *current* prefix. However, the final set is different).
    # Wait, the logic holds: if 1/k was a sum of reciprocals of {k+1..24} \ {some},
    # then adding k to the final set (which is a subset of {k+1..24}) would still be unsafe?
    # No, the final set is a subset of {k+1..24}.
    # If 1/k = sum(1/y for y in T) where T is a subset of {k+1..24},
    # and the final set S_final is a subset of {k+1..24},
    # it is possible that T is NOT a subset of S_final.
    # So k might be safe to add to S_final even if it was rejected during greedy.
    
    excluded = list(set(range(1, 25)) - best_set)
    # Sort excluded in decreasing order? Or increasing?
    # Let's try adding them one by one.
    # We should re-check validity of the WHOLE set when adding.
    
    # To be efficient, we can try to add excluded elements.
    # Since adding a smaller element is "riskier" (larger reciprocal), maybe add larger excluded first?
    # Or just iterate.
    
    # Let's try to add each excluded element if it keeps the set valid.
    # We need to check if adding x to best_set results in a valid set.
    # This requires checking all elements in best_set U {x}.
    # But we know best_set is valid.
    # So we only need to check:
    # 1. Is x safe? (1/x not sum of others in best_set)
    # 2. For each y in best_set, is y still safe?
    #    1/y not sum of subset of (best_set \ {y}) U {x}.
    #    Since best_set is valid, 1/y is not sum of subset of best_set \ {y}.
    #    The only new subsets are those containing x.
    #    So check: 1/y != 1/x + sum(subset of best_set \ {y, x}).
    #    => 1/y - 1/x != sum(subset of best_set \ {y, x}).
    #    Note: if 1/y - 1/x <= 0, it's impossible (sums are positive).
    #    So only need to check if 1/y > 1/x, i.e., y < x.
    #    Since we are adding x (which was excluded), and best_set contains larger numbers mostly,
    #    this check might be cheap.
    
    # Let's implement a helper to check if adding x to a valid set S is safe.
    def _can_add(x, S):
        # S is a set of integers, assumed valid.
        # Check if S U {x} is valid.
        # 1. Check x against S
        if not _is_safe(x, list(S)):
            return False
        # 2. Check y in S against S U {x}
        for y in S:
            if y >= x:
                # 1/y <= 1/x.
                # Any sum involving 1/x is >= 1/x >= 1/y.
                # Equality only if y=x (impossible) or sum has only 1/x (impossible, non-empty subset of others).
                # Actually, if y > x, 1/y < 1/x. Sum involving 1/x is > 1/x > 1/y.
                # If y == x, not in S.
                # So y is safe.
                continue
            
            # y < x, so 1/y > 1/x.
            # Check if 1/y - 1/x is a subset sum of S \ {y, x}.
            target_diff = Fraction(1, y) - Fraction(1, x)
            others = [z for z in S if z != y and z != x]
            if _is_safe(1 / target_diff if target_diff != 0 else 0, others): # This is wrong, _is_safe takes int
                # This logic is flawed. _is_safe checks if 1/candidate is a sum.
                # We need to check if target_diff is a sum of reciprocals of others.
                # Let's write a specific check.
                pass
        
        # Let's just use the full check for simplicity, as N is small (<=24).
        # But full check is O(N * 2^(N/2)).
        # With N=24, 2^12 = 4096. 24 * 4096 ~ 100k ops.
        # We might do this for each excluded element.
        # Excluded elements <= 24.
        # Total ~ 24 * 100k = 2.4M ops. Should be fast enough in Python?
        # Fraction arithmetic is slow.
        
        # Let's stick to the simpler strategy first: just try to add excluded elements
        # and verify the whole set is valid using _is_valid_full.
        # If valid, keep it.
        
        new_set = S | {x}
        return _is_valid_full(new_set)

    # Try to add excluded elements
    # Sort excluded in decreasing order?
    # Adding larger elements (smaller reciprocals) is generally safer.
    for x in sorted(excluded, reverse=True):
        if time.time() > deadline:
            break
        if _can_add(x, best_set):
            best_set.add(x)
    
    # Try to add remaining excluded elements (those not added in the first pass)
    # Maybe some combinations work?
    # Let's do a second pass.
    excluded = list(set(range(1, 25)) - best_set)
    for x in sorted(excluded, reverse=True):
        if time.time() > deadline:
            break
        if _can_add(x, best_set):
            best_set.add(x)
            
    # If we have time, try swapping?
    # Remove one element, add two?
    # This is complex. Let's see if we can improve the greedy.
    
    # The greedy from 24 down to 1 is the primary driver.
    # The "add back" step might recover some.
    
    # Let's also try a different greedy:
    # Start with empty, add 24, 23, ...
    # This is what we did.
    
    # What if we try to include 1?
    # 1 is very risky.
    
    # Let's return the best set found.
    if not best_set:
        return [24]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
