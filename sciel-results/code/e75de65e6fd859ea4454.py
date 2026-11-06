"""
Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256].
Combines bitmask-based fast AP checking with multi-step local search (1-swap, 2-swap).
"""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

N = 256
TIME_LIMIT = 58.0  # Safety margin before 60s kill

def precompute_conflicts():
    """
    Precompute for each x in [1, N], the list of pairs (a, b) such that {x, a, b} is a 3-AP.
    Specifically, for a candidate x, if we want to add x to S, we must ensure that 
    for no pair (a, b) in conflicts[x], both a and b are in S.
    
    A 3-AP involving x can be:
    1. x, a, b where x is the first term: b = 2a - x. Pairs: (a, 2a-x).
    2. a, x, b where x is the middle term: b = 2x - a. Pairs: (a, 2x-a).
    3. a, b, x where x is the last term: a = 2b - x. Pairs: (2b-x, b).
    
    We can unify this: for any a in S, check if 2a - x is in S (covers x as min or max relative to a, 
    wait, if x is min: x, a, 2a-x. If x is max: 2a-x, a, x. Both imply {a, 2a-x} is a pair).
    If x is mid: a, x, 2x-a. This implies {a, 2x-a} is a pair.
    
    So, for a given x, the "forbidden pairs" are:
    - For all a in [1, N] such that 2a - x is in [1, N] and 2a - x != a and 2a - x != x:
      Pair (a, 2a - x). Note: if 2a-x == x, then a=x, which is not in S (x is new). 
      If 2a-x == a, then a=x, same.
    - For all a in [1, N] such that 2x - a is in [1, N] and 2x - a != a and 2x - a != x:
      Pair (a, 2x - a).
      
    Actually, the set of pairs {a, b} such that {x, a, b} is a 3-AP is:
    1. a + b = 2x (x is middle). Pairs (a, 2x-a).
    2. a + x = 2b => b = (a+x)/2 (x is endpoint, b is middle). 
       This is equivalent to checking if (a+x) is even and (a+x)/2 is in S.
       Wait, the standard check "is x compatible with S" is:
       - No a, b in S such that a+b=2x.
       - No a, b in S such that a+x=2b => b=(a+x)/2.
       - No a, b in S such that x+b=2a => a=(x+b)/2.
       
    Using bitmasks:
    Let M be the bitmask of S.
    1. Check if there exist a, b in S with a+b=2x.
       This is equivalent to: Is there an a in S such that (2x-a) is in S?
       We can iterate a in S? No, we want O(1) or fast.
       
    Alternative approach for fast check:
    Precompute for each x, a list of "partner" pairs.
    If we store S as a bitmask M:
    - Check 1 (x is middle): 
      We need to know if there is a pair (a, b) in S such that a+b=2x.
      We can precompute for each x, the list of a such that b=2x-a is valid.
      Then check if any (a, 2x-a) are both in M.
      
    - Check 2 (x is endpoint):
      We need to know if there is a pair (a, b) in S such that a+b=2x is NOT the case, 
      but rather one of them is x? No, x is the new element.
      If x is the min: x, a, b. Then b = 2a - x.
      So we need to check if there exists a in S such that (2a - x) is in S.
      
    So, for a given x, we need to check two sets of pairs:
    P1 = { (a, 2x-a) for a in [1, N] if 1 <= 2x-a <= N and 2x-a != a and 2x-a != x }
    P2 = { (a, 2a-x) for a in [1, N] if 1 <= 2a-x <= N and 2a-x != a and 2a-x != x }
    
    Note: If (a, b) is in P1, then a+b=2x.
    If (a, b) is in P2, then a+b=2x? No. 2a-x=b => a+b = 2a-x + b? No.
    If b = 2a - x, then x, a, b is an AP. The sum x+b = 2a. So the "middle" is a.
    So P2 checks if there is a 'middle' a in S such that the other endpoint b is in S.
    
    So the condition for x to be addable to S is:
    1. No pair (a, b) in P1 is fully contained in S.
    2. No pair (a, b) in P2 is fully contained in S.
    
    We can precompute these pairs for each x.
    """
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # P1: x is middle. a + b = 2x.
        for a in range(1, N + 1):
            b = 2 * x - a
            if 1 <= b <= N and b != a and b != x:
                # We can just store (min(a,b), max(a,b)) to avoid duplicates if needed, 
                # but for bitmask check, order doesn't matter as long as we check both bits.
                conflicts[x].append((a, b))
        
        # P2: x is endpoint. x, a, b => b = 2a - x.
        # This means a is the middle.
        for a in range(1, N + 1):
            b = 2 * a - x
            if 1 <= b <= N and b != a and b != x:
                # Here a is the middle. The pair of elements in S that conflict with x is (a, b).
                # Wait, if a is in S and b is in S, and x is added, then {x, a, b} is an AP.
                # So yes, if both a and b are in S, x cannot be added.
                conflicts[x].append((a, b))
                
    return conflicts

# Precompute once
CONFLICTS = precompute_conflicts()

def can_add(x, mask):
    """
    Check if x can be added to the set represented by mask.
    mask is an integer where bit i (0-indexed) represents number i+1.
    """
    # x is 1-indexed. Bit index is x-1.
    # If x is already in mask, return False (though we shouldn't call this if x is in S)
    if (mask >> (x - 1)) & 1:
        return False
        
    for a, b in CONFLICTS[x]:
        # Check if both a and b are in mask
        # Bit a is at index a-1. Bit b is at index b-1.
        if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
            return False
    return True

def reciprocal_sum_from_mask(mask):
    """Compute exact reciprocal sum from bitmask."""
    total = Fraction(0, 1)
    for i in range(1, N + 1):
        if (mask >> (i - 1)) & 1:
            total += Fraction(1, i)
    return total

def mask_to_set(mask):
    """Convert bitmask to a set of integers."""
    return {i + 1 for i in range(N) if (mask >> i) & 1}

def greedy_construct_bitmask():
    """Greedy construction using bitmask."""
    mask = 0
    # We want to maximize sum of 1/x. 
    # Greedy by smallest x first is a good heuristic for reciprocal sums because 1/x decreases.
    for x in range(1, N + 1):
        if can_add(x, mask):
            mask |= (1 << (x - 1))
    return mask

def local_search_bitmask(initial_mask, time_limit):
    """
    Local search using bitmask representation.
    Moves:
    1. Add a single element (if valid).
    2. Swap one element for another (remove one, add one).
    3. Swap two elements for two others (remove two, add two).
    """
    start_time = time.time()
    current_mask = initial_mask
    current_score = reciprocal_sum_from_mask(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Precompute weights for quick float comparison (for speed in inner loops, final check with Fraction)
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Helper to get set of elements in mask
    def get_elements(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
    
    # Helper to get set of candidates (not in mask)
    def get_candidates(mask):
        return [i + 1 for i in range(N) if not ((mask >> i) & 1)]

    # Phase 1: Add-only (Greedy completion)
    # This is fast and finds any obvious missing small numbers.
    while True:
        if time.time() - start_time > time_limit:
            break
        improved = False
        candidates = get_candidates(current_mask)
        # Sort by value ascending (smaller x -> larger weight)
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if can_add(x, current_mask):
                current_mask |= (1 << (x - 1))
                current_score += Fraction(1, x)
                improved = True
                if current_score > best_score:
                    best_mask = current_mask
                    best_score = current_score
        if not improved:
            break

    # Phase 2: 1-Swap (Remove one, add one)
    # We try to remove larger elements (lower weight) and add smaller ones (higher weight).
    while True:
        if time.time() - start_time > time_limit:
            break
        improved = False
        elements = get_elements(current_mask)
        # Sort elements by value descending (largest x first, as they have lowest weight and are most removable)
        elements.sort(reverse=True)
        
        for x_out in elements:
            if time.time() - start_time > time_limit:
                break
            
            mask_without_out = current_mask & ~(1 << (x_out - 1))
            score_without_out = current_score - Fraction(1, x_out)
            
            # Try to add any candidate y
            candidates = get_candidates(mask_without_out)
            # Sort candidates by value ascending (smallest y first, highest weight)
            for y in candidates:
                if time.time() - start_time > time_limit:
                    break
                if y == x_out:
                    continue
                if can_add(y, mask_without_out):
                    new_score = score_without_out + Fraction(1, y)
                    if new_score > current_score:
                        current_mask = mask_without_out | (1 << (y - 1))
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_mask = current_mask
                            best_score = current_score
                        break # Found an improvement, restart loop or continue? Restarting is safer for local search.
            
        if not improved:
            break

    # Phase 3: 2-Swap (Remove two, add two)
    # This is more expensive but can escape local optima of 1-swap.
    # We limit the depth or number of attempts to stay within time.
    while True:
        if time.time() - start_time > time_limit:
            break
        improved = False
        elements = get_elements(current_mask)
        elements.sort(reverse=True) # Remove largest
        
        # To keep it manageable, we don't try ALL pairs of removals if the set is large.
        # We try a subset of pairs involving the largest elements.
        # Or we can iterate through all pairs if the set is small enough.
        # Max size of 3-AP-free set in [1, 256] is around 100-120? 
        # Pairs of 100 elements is ~5000. For each pair, checking candidates is O(N * deg).
        # This might be too slow if done naively in Python.
        # We will iterate through pairs but break early if time is tight.
        
        for i in range(len(elements)):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, len(elements)):
                if time.time() - start_time > time_limit:
                    break
                
                x1 = elements[i]
                x2 = elements[j]
                
                mask_without = current_mask & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                score_without = current_score - Fraction(1, x1) - Fraction(1, x2)
                
                candidates = get_candidates(mask_without)
                # We need to find two distinct y, z from candidates such that:
                # 1. y can be added to mask_without.
                # 2. z can be added to mask_without | (1 << (y-1)).
                # 3. Score improves.
                
                # Optimization: Only consider candidates y that are "small" (high weight).
                # Sort candidates
                candidates.sort()
                
                found_improvement = False
                for y in candidates:
                    if time.time() - start_time > time_limit:
                        break
                    if y == x1 or y == x2:
                        continue
                    if not can_add(y, mask_without):
                        continue
                    
                    temp_mask = mask_without | (1 << (y - 1))
                    temp_score = score_without + Fraction(1, y)
                    
                    for z in candidates:
                        if time.time() - start_time > time_limit:
                            break
                        if z <= y: # Avoid duplicates and z==y
                            continue
                        if z == x1 or z == x2:
                            continue
                        if not can_add(z, temp_mask):
                            continue
                        
                        new_score = temp_score + Fraction(1, z)
                        if new_score > current_score:
                            current_mask = temp_mask | (1 << (z - 1))
                            current_score = new_score
                            improved = True
                            found_improvement = True
                            if current_score > best_score:
                                best_mask = current_mask
                                best_score = current_score
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break # Restart 2-swap loop with new state
            
        if not improved:
            break

    return best_mask

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    greedy_mask = greedy_construct_bitmask()
    greedy_score = reciprocal_sum_from_mask(greedy_mask)
    
    # Phase 2: Local search
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 1.0:
        improved_mask = local_search_bitmask(greedy_mask, remaining_time - 0.5)
        improved_score = reciprocal_sum_from_mask(improved_mask)
        if improved_score > greedy_score:
            greedy_mask = improved_mask
            greedy_score = improved_score
    
    # Convert to sorted list
    result = mask_to_set(greedy_mask)
    return sorted(result)

# EVOLVE-BLOCK-END
