"""Rewritten program using a search approach to maximize set size under the divisibility constraint."""
import time
import random
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0  # safety margin before 90s wall-clock kill

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _valid_set(elements):
    """
    Check if a set of distinct integers satisfies the constraint:
    No two distinct nonempty subset sums may divide one another.
    (Equal sums are forbidden because they imply divisibility via 2s and s.)
    """
    elements = list(elements)
    n = len(elements)
    if n == 0:
        return True
    
    # For n <= 20, brute force sums
    if n <= 20:
        current_sums = [0]
        for e in elements:
            current_sums = current_sums + [x + e for x in current_sums]
        
        # Check for duplicates
        if len(current_sums) != len(set(current_sums)):
            return False
        
        nonempty_sums = [s for s in current_sums if s > 0]
        nonempty_sums.sort()
        for i, s in enumerate(nonempty_sums):
            for t in nonempty_sums[i+1:]:
                if t % s == 0:
                    return False
        return True
    else:
        # For larger n, use the incremental builder's logic
        # This is a fallback; the main search uses incremental building
        return False

def _try_add_element_fast(current_elements, candidate, current_sums_set, current_sums_list, forbidden_multiples):
    """
    Try to add 'candidate' to 'current_elements'.
    Uses forbidden_multiples to quickly check divisibility.
    Returns (new_elements, new_sums_set, new_sums_list, new_forbidden_multiples) if valid, else None.
    """
    # Check for collisions: new sum s + candidate must not be in current_sums_set
    new_sums = []
    for s in current_sums_list:
        ns = s + candidate
        if ns in current_sums_set:
            return None  # Collision
        new_sums.append(ns)
    
    # Check divisibility using forbidden_multiples
    # forbidden_multiples contains all values that are multiples of existing sums
    # A new sum t is invalid if t is in forbidden_multiples (some existing sum divides t)
    # OR if some new sum divides t (we need to check new vs new separately)
    
    # First, check if any new sum is a multiple of an existing sum
    for t in new_sums:
        if t in forbidden_multiples:
            return None  # Some existing sum divides t
    
    # Check new sums against each other
    # Sort new sums
    new_sums_sorted = sorted(new_sums)
    for i in range(len(new_sums_sorted)):
        s1 = new_sums_sorted[i]
        for j in range(i + 1, len(new_sums_sorted)):
            s2 = new_sums_sorted[j]
            if s2 % s1 == 0:
                return None
    
    # Check if any existing sum is a multiple of a new sum
    # This means: for some new sum t, t divides some existing sum s
    # Equivalently: s is in the set of multiples of t
    # We can check: for each new sum t, check if any existing sum s satisfies s % t == 0
    # But this is expensive. Alternative: check if any existing sum is in the multiples of t
    # We don't have a precomputed set for this direction.
    # However, note that if t divides s, then s is a multiple of t.
    # We can precompute for each new sum t, the set of multiples of t that are in current_sums_set.
    # But that's also expensive.
    
    # Alternative approach: since we're adding one element at a time, and the set is small,
    # we can just check directly. But let's optimize:
    # For each new sum t, check if there exists s in current_sums_set such that s % t == 0.
    # This is equivalent to: t divides some s in current_sums_set.
    # We can check by iterating through current_sums_set and checking s % t == 0.
    # But current_sums_set can be large.
    
    # Better: precompute for the new sums.
    # For each new sum t, the multiples of t that could be in current_sums_set are:
    # t, 2t, 3t, ... up to max(current_sums_set)
    # We can check if any of these are in current_sums_set.
    
    max_existing_sum = max(current_sums_set) if current_sums_set else 0
    for t in new_sums:
        if t <= 0:
            continue
        # Check multiples of t up to max_existing_sum
        multiple = t
        while multiple <= max_existing_sum:
            if multiple in current_sums_set:
                return None  # t divides multiple, which is an existing sum
            multiple += t
    
    # If we passed all checks, it's valid.
    new_sums_set = current_sums_set.union(set(new_sums))
    new_sums_list = current_sums_list + new_sums
    
    # Update forbidden_multiples: add multiples of new sums
    # But we only need to track multiples up to some limit (max possible sum)
    # Max possible sum is sum of all elements, which is at most 256*30 = 7680
    # But we can be more conservative: track multiples up to 256*30 = 7680
    MAX_SUM = 7680
    new_forbidden = set(forbidden_multiples)
    for t in new_sums:
        if t <= 0:
            continue
        multiple = t
        while multiple <= MAX_SUM:
            new_forbidden.add(multiple)
            multiple += t
    
    return (current_elements + [candidate], new_sums_set, new_sums_list, new_forbidden)

def construct():
    start_time = time.time()
    best_set = []
    best_sums_set = {0}
    best_sums_list = [0]
    best_forbidden = set()
    
    # Precompute primes for candidate ordering
    primes = [i for i in range(1, 257) if _is_prime(i)]
    
    max_iterations = 2000  # Number of restarts
    
    for restart in range(max_iterations):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        # Choose a strategy for this restart
        strategy = restart % 6
        
        if strategy == 0:
            # Random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif strategy == 1:
            # Descending
            candidates = list(range(256, 0, -1))
        elif strategy == 2:
            # Primes first, then random
            candidates = primes + [i for i in range(1, 257) if i not in primes]
            random.shuffle(candidates[len(primes):])
        elif strategy == 3:
            # Large numbers with high probability
            candidates = []
            for i in range(1, 257):
                if random.random() < (i / 256) ** 2:
                    candidates.append(i)
            if not candidates:
                candidates = [256]
        elif strategy == 4:
            # Coprime-focused: prefer numbers that are coprime to many others
            # Heuristic: prefer primes and numbers with few divisors
            candidates = sorted(range(1, 257), key=lambda x: -x if _is_prime(x) else 0)
            random.shuffle(candidates)
        else:
            # Adaptive: mix of strategies
            if random.random() < 0.5:
                candidates = list(range(1, 257))
                random.shuffle(candidates)
            else:
                candidates = primes + [i for i in range(1, 257) if i not in primes]
                random.shuffle(candidates)
        
        current_elements = []
        current_sums_set = {0}
        current_sums_list = [0]
        current_forbidden = set()
        
        # Greedy pass
        added_any = True
        while added_any:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            added_any = False
            for cand in candidates:
                if time.time() - start_time > DEADLINE_SECONDS - 5:
                    break
                
                if cand in current_elements:
                    continue
                
                # Try to add
                result = _try_add_element_fast(
                    current_elements, cand, current_sums_set, current_sums_list, current_forbidden
                )
                
                if result is not None:
                    current_elements, current_sums_set, current_sums_list, current_forbidden = result
                    added_any = True
                    
                    # Update best if better
                    if len(current_elements) > len(best_set):
                        best_set = current_elements.copy()
                        best_sums_set = current_sums_set
                        best_sums_list = current_sums_list
                        best_forbidden = current_forbidden
                    break  # Restart the candidate loop with the updated state
        
        # End of greedy pass for this restart
        if len(current_elements) > len(best_set):
            best_set = current_elements.copy()
            best_sums_set = current_sums_set
            best_sums_list = current_sums_list
            best_forbidden = current_forbidden

    # Final verification of best_set
    best_set = list(set(best_set))
    best_set.sort()
    
    if not best_set:
        best_set = [31, 32, 34]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
