import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Strategy: Direct construction using known admissible tuple structure
    # Greedy repair approach.
    
    num_restarts = 100 # Reduced restarts, focus on optimization per restart
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Start with all numbers
        current_set = list(range(N))
        current_set_set = set(current_set)
        
        # Greedy Repair Loop
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime
            violating_p = None
            for p in primes:
                if p > k:
                    break
                # Check if all residues mod p are covered
                seen = bytearray(p)
                count = 0
                for v in current_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = 1
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                # Admissible
                break
            
            # We need to remove elements to break the coverage for violating_p
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            # Find the class with the minimum number of elements
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            # Randomly pick from the top few smallest classes for diversity
            if random.random() < 0.8:
                min_idx = class_sizes[0][1]
            else:
                top_k = min(5, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            # Remove all elements in this class
            to_remove = set(classes[min_idx])
            current_set_set -= to_remove
            current_set = list(current_set_set)
        
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        # Optimization: Fast local search to add back elements
        # Precompute empty classes for current set
        k = len(current_set_set)
        if k == 0: continue
        
        # Determine relevant primes: primes <= k
        relevant_primes = [p for p in primes if p <= k]
        
        # Compute empty_classes: dict p -> set of residues r such that no x in S has x%p == r
        empty_classes = {}
        for p in relevant_primes:
            seen = bytearray(p)
            for v in current_set_set:
                seen[v % p] = 1
            empty = set()
            for r in range(p):
                if not seen[r]:
                    empty.add(r)
            empty_classes[p] = empty
            
        # Check if current set is valid (it should be)
        for p in relevant_primes:
            if not empty_classes[p]:
                break # Should not happen
        
        # Local Search: Try to add elements back
        candidates = [x for x in range(N) if x not in current_set_set]
        random.shuffle(candidates)
        
        max_add_passes = 50
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            added_any = False
            # Iterate through all candidates
            for x in candidates:
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                
                new_size = k + 1
                # Check if we can add x
                can_add = True
                
                # 1. Check existing primes p <= k
                for p in relevant_primes:
                    r_x = x % p
                    if r_x in empty_classes[p]:
                        # Adding x fills this empty class
                        # If this was the ONLY empty class, it becomes violated
                        if len(empty_classes[p]) == 1:
                            can_add = False
                            break
                
                if not can_add:
                    continue
                
                # 2. Check if new_size introduces a new prime constraint
                if new_size in [p for p in primes if p <= new_size]: # new_size is prime?
                    # Actually, we need to check primes <= new_size.
                    # The only new prime is new_size if it is prime.
                    if new_size in primes: # List membership is slow, use set?
                         pass # We'll handle this below
                
                # Better: Check if new_size is prime
                is_new_prime = (new_size > 1 and all(new_size % i != 0 for i in range(2, int(new_size**0.5)+1)))
                
                if is_new_prime:
                    p_new = new_size
                    # We need an empty class mod p_new in the NEW set.
                    # Since we are adding x, the empty classes in the new set are:
                    # (old empty classes mod p_new) - {x % p_new}
                    # Wait, we didn't track empty classes for p_new before.
                    # We need to compute if there is ANY empty class mod p_new in current_set_set.
                    # If current_set_set has NO empty class mod p_new, then adding x (which doesn't create empty classes)
                    # will result in NO empty class mod p_new. Violation.
                    # If current_set_set HAS an empty class mod p_new:
                    #   Let E be the set of empty residues mod p_new in current_set_set.
                    #   In new set, empty residues are E - {x % p_new}.
                    #   We need E - {x % p_new} to be non-empty.
                    #   So we need len(E) > 1 if x%p_new in E, or len(E) >= 1 if x%p_new not in E.
                    
                    seen_new = bytearray(p_new)
                    for v in current_set_set:
                        seen_new[v % p_new] = 1
                    empty_new = set()
                    for r in range(p_new):
                        if not seen_new[r]:
                            empty_new.add(r)
                    
                    if not empty_new:
                        can_add = False
                    else:
                        r_x_new = x % p_new
                        if r_x_new in empty_new:
                            if len(empty_new) == 1:
                                can_add = False
                        # else can_add remains True (empty_new still has others)
                
                if can_add:
                    current_set_set.add(x)
                    k += 1
                    added_any = True
                    # Update empty_classes for future checks in this pass?
                    # This is tricky because relevant_primes might change if k increases past a prime.
                    # For simplicity, we can restart the pass or just accept that empty_classes is stale.
                    # Stale empty_classes:
                    # If we add x, empty_classes[p] might shrink.
                    # Our check `if len(empty_classes[p]) == 1` might be wrong if it was 1 and we filled it?
                    # No, if we filled it, can_add would be False.
                    # So if can_add is True, empty_classes[p] is NOT shrunk to 0.
                    # It might shrink from 2 to 1.
                    # So the stored `empty_classes[p]` is now too large (contains a residue that is no longer empty).
                    # This might allow us to add another element that fills the *other* empty class,
                    # thinking there are 2, but there is only 1 (the one we just filled? No, we didn't fill it).
                    # Wait.
                    # If empty_classes[p] = {0, 1}.
                    # Add x with x%p=0.
                    # can_add check: 0 in {0,1}, len=2 > 1. OK.
                    # New empty_classes[p] should be {1}.
                    # Stored is {0,1}.
                    # Next candidate y with y%p=1.
                    # Check: 1 in {0,1}, len=2 > 1. OK.
                    # But actually 0 is filled, 1 is about to be filled.
                    # We should have rejected y.
                    # So we MUST update empty_classes.
                    
                    # Update empty_classes for all p <= k
                    # This is expensive.
                    # Alternative: Don't update, but re-verify admissibility of the whole set periodically?
                    # Or just break the pass and restart with fresh empty_classes.
                    
                    # Let's break the inner loop and restart the pass.
                    break
            
            if not added_any:
                break
            
            # If we added, we broke the loop. 
            # We should restart the candidate loop with updated state.
            # To be efficient, we can just continue the outer pass loop,
            # but we need to recompute empty_classes.
            # Recomputing empty_classes is O(K * pi(K)).
            # This is acceptable if we do it once per pass.
            
            # Recompute empty_classes
            relevant_primes = [p for p in primes if p <= k]
            empty_classes = {}
            for p in relevant_primes:
                seen = bytearray(p)
                for v in current_set_set:
                    seen[v % p] = 1
                empty = set()
                for r in range(p):
                    if not seen[r]:
                        empty.add(r)
                empty_classes[p] = empty

        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
            
    return list(best_set)

# Initial call to set start time and warm up
construct()
