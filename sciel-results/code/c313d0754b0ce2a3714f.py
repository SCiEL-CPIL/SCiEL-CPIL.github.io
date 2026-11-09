import time
import random
from fractions import Fraction
from math import isqrt

N = 256

def is_valid_addition(S_set, x):
    """
    Check if adding x to S_set results in a set with no nonconstant 3-APs.
    S_set is a set of integers. x is an integer not in S_set.
    """
    # If x is added, it forms an AP with u, v in S_set if:
    # 1. u, x, v are in AP => 2x = u + v => v = 2x - u
    # 2. x, u, v are in AP => 2u = x + v => v = 2u - x
    # 3. u, v, x are in AP => 2v = u + x => v = (u+x)/2 (u+x must be even)
    
    # We can just check for any u in S_set:
    # - If (2*x - u) is in S_set, we have AP (u, x, 2*x-u) or (2*x-u, x, u).
    #   Note: If 2*x-u == x, then u=x, which is false.
    # - If (2*u - x) is in S_set, we have AP (2*u-x, u, x) or (x, u, 2*u-x).
    #   Note: If 2*u-x == x, then u=x, false.
    #   Note: If 2*u-x == u, then u=x, false.
    
    # Actually, checking `2*x - u in S_set` covers the case where x is the middle term.
    # Checking `2*u - x in S_set` covers the case where x is an endpoint (and u is the other endpoint or middle?).
    # Let's be precise.
    # An AP is a, b, c with a+c=2b.
    # If x is in the AP, it can be a, b, or c.
    # Case 1: x is middle. a, x, c. a+c=2x. c=2x-a. For any a in S, if c in S, violation.
    # Case 2: x is first. x, b, c. x+c=2b. c=2b-x. For any b in S, if c in S, violation.
    # Case 3: x is last. a, b, x. a+x=2b. a=2b-x. For any b in S, if a in S, violation.
    
    # So we need to check:
    # 1. For any a in S, is (2*x - a) in S?
    # 2. For any b in S, is (2*b - x) in S?
    
    # Note: If we check 1, do we cover 2 and 3?
    # In Case 1, x is middle.
    # In Case 2, x is first. The AP is x, b, c. Here b is middle.
    #   This is covered by checking if there exists a middle term b in S such that the other endpoint c is in S.
    #   The other endpoint is c = 2b - x.
    #   So checking `2*b - x in S` for all b in S covers Case 2 (x first) and Case 3 (x last, by symmetry of b).
    #   Wait.
    #   If x is last: a, b, x. b is middle. a = 2b - x. Check if a in S.
    #   If x is first: x, b, c. b is middle. c = 2b - x. Check if c in S.
    #   So `for b in S: if (2*b - x) in S: return False` covers x being an endpoint.
    
    # If x is middle: a, x, c. x is middle. a = 2x - c or c = 2x - a.
    #   So `for a in S: if (2*x - a) in S: return False` covers x being the middle.
    
    # So the full check is:
    for u in S_set:
        # x is middle
        v = 2 * x - u
        if v in S_set:
            return False
        # x is endpoint (u is middle)
        v = 2 * u - x
        if v in S_set:
            return False
            
    return True

def greedy_construct(order):
    S = []
    S_set = set()
    for x in order:
        if is_valid_addition(S_set, x):
            S.append(x)
            S_set.add(x)
    return S, S_set

def local_search(S, S_set, time_limit=60.0):
    # S is list, S_set is set
    # We want to swap y in S with x not in S, where x < y (to increase weight)
    # Actually, any swap that increases weight is good.
    # Weight change = 1/x - 1/y. We want 1/x > 1/y => x < y.
    
    # Precompute list of non-selected elements
    all_elements = set(range(1, N + 1))
    ns_list = list(all_elements - S_set)
    
    # Sort ns_list by value? We prefer small x.
    # Sort S by value? We prefer removing large y.
    # Let's iterate all pairs (x, y) with x < y.
    
    # To speed up, we can sort S descending and ns_list ascending.
    # But checking all pairs is O(|S| * |ns|).
    # |S| ~ 100, |ns| ~ 150. 15,000 checks.
    # Each check is O(|S|) ~ 100.
    # Total 1.5M ops per pass.
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2.0:
            break
            
        # Try all swaps x, y with x < y
        # We can optimize by iterating x in ns_list (small to large) and y in S (large to small)
        # If 1/x - 1/y > 0, i.e., x < y, it's a potential gain.
        
        # To avoid O(N^2) overhead in Python, we might just iterate.
        # Let's try to be smart.
        # We want to find ANY improving swap.
        
        for x in ns_list:
            # If x is small, 1/x is large.
            # We want to swap it with a large y.
            # Iterate y in S descending?
            # If we find one valid swap, we do it and restart the loop?
            # Or do we accumulate?
            # Standard hill climbing: find one improving move, apply it, restart.
            
            # To make it faster, let's just check if we can add x by removing y.
            # We need to check validity of (S - {y}) U {x}.
            
            # Optimization: 
            # If we remove y, the set becomes S' = S - {y}.
            # Then we check if x can be added to S'.
            # is_valid_addition(S', x)
            
            # This is still O(|S|) per check.
            
            pass # Placeholder for logic below

    # Simpler Local Search Implementation:
    # Iterate through all possible (x, y) pairs where x not in S, y in S, and x < y.
    # Check if swap is valid.
    # If yes, apply and restart.
    
    S_list = list(S)
    # Sort S_list descending to try removing large elements first?
    # Sort ns_list ascending to try adding small elements first?
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2.0:
            break
        
        # Generate candidates
        # We want x < y.
        # Let's iterate x in ns_list.
        # And y in S_list.
        
        # To speed up, we can break early if 1/x <= 1/y (i.e., x >= y)
        # But we need to check validity first.
        
        for x in ns_list:
            # If x is already in S, skip (shouldn't happen)
            if x in S_set:
                continue
                
            # We want to remove some y > x
            # Iterate y in S_list
            for y in S_list:
                if y <= x:
                    continue # We want x < y for weight increase
                
                # Check if swap is valid
                # New set: S_new = (S_set - {y}) U {x}
                # Check if x is valid to add to S_set - {y}
                
                # Efficient check:
                # x creates AP with S_new iff:
                # 1. There exist u, v in S_new such that u, x, v is AP.
                # 2. There exist u, v in S_new such that x, u, v is AP.
                
                # Note that y is removed.
                # So we check conflicts with S_set \ {y}.
                
                valid = True
                for u in S_set:
                    if u == y:
                        continue
                    
                    # x is middle: u, x, 2x-u
                    v = 2 * x - u
                    if v in S_set and v != y: # v must be in S_new
                        valid = False
                        break
                    
                    # x is endpoint: u is middle. 2u-x
                    v = 2 * u - x
                    if v in S_set and v != y: # v must be in S_new
                        valid = False
                        break
                        
                    if not valid:
                        break
                    
                if valid:
                    # Perform swap
                    S_set.remove(y)
                    S_set.add(x)
                    S_list.remove(y)
                    S_list.append(x)
                    ns_list.remove(x)
                    ns_list.append(y)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                break # Restart outer loop too? 
                # Actually, breaking inner loop is enough to restart the 'while improved' if we structure it right.
                # But we broke out of 'for x'. We need to break out of 'for x' as well.
                # The 'break' above breaks 'for y'.
                # We need to break 'for x'.
                # Let's use a flag or return.
                # Actually, just break out of both.
                break

    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_set = None
    best_weight = Fraction(0, 1)
    
    # 1. Generate initial sets
    initial_sets = []
    
    # Greedy Ascending
    s1, s1_set = greedy_construct(list(range(1, N + 1)))
    initial_sets.append((s1_set, sum(Fraction(1, x) for x in s1_set)))
    
    # Greedy Descending
    s2, s2_set = greedy_construct(list(range(N, 0, -1)))
    initial_sets.append((s2_set, sum(Fraction(1, x) for x in s2_set)))
    
    # Random Greedy (a few trials)
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s, s_set = greedy_construct(order)
        w = sum(Fraction(1, x) for x in s_set)
        initial_sets.append((s_set, w))
        
    # 2. Local Search on each initial set
    for S_init, w_init in initial_sets:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
            
        # Copy set to avoid modifying original if we want to keep it?
        # Local search modifies the set passed.
        # We pass a copy.
        S_copy = set(S_init)
        
        S_improved = local_search(S_copy, S_copy, time_limit=deadline - start_time - 2.0)
        
        w_improved = sum(Fraction(1, x) for x in S_improved)
        
        if w_improved > best_weight:
            best_weight = w_improved
            best_set = S_improved

    # 3. If no set found (shouldn't happen), fallback
    if not best_set:
        best_set = {1}
        
    return list(best_set)

# EVOLVE-BLOCK-END
