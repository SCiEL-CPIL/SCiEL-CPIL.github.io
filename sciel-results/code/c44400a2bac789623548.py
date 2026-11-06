from fractions import Fraction
import time
import random

def construct():
    start_time = time.time()
    timeout = 55
    
    best_S = set()
    
    # Deterministic orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
    ]
    
    # Add some random orders
    while time.time() - start_time < timeout - 10:
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        orders.append(random_order)
        
    for order in orders:
        if time.time() - start_time > timeout:
            break
        S = _greedy_incremental(order, timeout - (time.time() - start_time))
        if len(S) > len(best_S):
            best_S = S
            
    # Local search: try to add any missing element
    if time.time() - start_time < timeout:
        # Reconstruct sums for best_S to allow further additions?
        # Or just use the slow _is_valid for local search?
        # Since best_S is large, slow _is_valid might be risky.
        # Let's try to rebuild state for best_S and then try adding.
        
        # Rebuild state
        S = set(best_S)
        sums = {Fraction(0)}
        sums_without = {}
        # We need to rebuild sums_without for best_S
        # This is expensive O(k 2^k).
        # If k is ~15, it's fine.
        if len(S) < 18: # Safety check
             sums, sums_without = _rebuild_state(S)
             
             for a in range(1, 25):
                 if time.time() - start_time > timeout:
                     break
                 if a in S:
                     continue
                 
                 # Check validity
                 valid = True
                 if Fraction(1, a) in sums:
                     valid = False
                 else:
                     for b in S:
                         target = Fraction(1, b) - Fraction(1, a)
                         if target in sums_without[b]:
                             valid = False
                             break
                 
                 if valid:
                     # Add a
                     S.add(a)
                     # Update state
                     inv_a = Fraction(1, a)
                     old_sums = sums
                     sums = set(old_sums)
                     for s in old_sums:
                         sums.add(s + inv_a)
                     
                     for b in list(S):
                         if b == a:
                             continue
                         old_sw = sums_without[b]
                         new_sw = set(old_sw)
                         for s in old_sw:
                             new_sw.add(s + inv_a)
                         sums_without[b] = new_sw
                     
                     sums_without[a] = old_sums
                     
                     if len(S) > len(best_S):
                         best_S = S

    return sorted(best_S)

def _greedy_incremental(order, time_limit):
    S = set()
    sums = {Fraction(0)}
    sums_without = {}
    start = time.time()
    
    for a in order:
        if time.time() - start > time_limit:
            break
        if a in S:
            continue
            
        # Check validity
        valid = True
        inv_a = Fraction(1, a)
        if inv_a in sums:
            valid = False
        else:
            for b in S:
                target = Fraction(1, b) - inv_a
                if target in sums_without[b]:
                    valid = False
                    break
        
        if valid:
            S.add(a)
            
            # Update sums
            old_sums = sums
            sums = set(old_sums)
            for s in old_sums:
                sums.add(s + inv_a)
                
            # Update sums_without
            for b in list(S):
                if b == a:
                    continue
                old_sw = sums_without[b]
                new_sw = set(old_sw)
                for s in old_sw:
                    new_sw.add(s + inv_a)
                sums_without[b] = new_sw
            
            sums_without[a] = old_sums

    return S

def _rebuild_state(S):
    S = set(S)
    sums = {Fraction(0)}
    for a in S:
        inv_a = Fraction(1, a)
        old_sums = sums
        sums = set(old_sums)
        for s in old_sums:
            sums.add(s + inv_a)
            
    sums_without = {}
    # For each b in S, compute subset sums of S \ {b}
    # We can do this by removing b from the full sum set? No, subset sums don't work that way.
    # We must recompute for each b.
    for b in S:
        others = [x for x in S if x != b]
        sw = {Fraction(0)}
        for x in others:
            inv_x = Fraction(1, x)
            old_sw = sw
            sw = set(old_sw)
            for s in old_sw:
                sw.add(s + inv_x)
        sums_without[b] = sw
        
    return sums, sums_without
