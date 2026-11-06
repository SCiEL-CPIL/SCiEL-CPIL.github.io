"""
Dissociated reciprocal set finder.
Strategy:
1. Use exact integer arithmetic via LCM scaling to avoid floating-point errors.
2. Efficient dissociation check with early exit.
3. Multi-phase search:
   a. Greedy construction with multiple orderings.
   b. Local search (additions, swaps, removals+re-adds).
   c. Random restarts with simulated annealing-like acceptance.
4. Time management with explicit deadline guard.
"""

import time
import math
import random
import sys

def _is_dissociated(S):
    """
    Check if set S has distinct reciprocal subset sums.
    Uses LCM scaling to convert to integer weights.
    """
    if not S:
        return True
    if len(S) == 1:
        return True
    
    # Compute LCM of all elements in S
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Compute integer weights
    weights = [L // x for x in S]
    
    # Efficient early-exit check using a set of seen sums
    # The condition is that all 2^k subset sums are distinct.
    # This is equivalent to: for each weight w, the new sums (old_sum + w) 
    # must not collide with any existing sum.
    seen = {0}
    for w in weights:
        new_sums = []
        # Check for collisions
        for s in seen:
            ns = s + w
            if ns in seen:
                return False
            new_sums.append(ns)
        # If we reach here, no collisions with existing sums.
        # Since s are unique, s+w are unique among themselves.
        seen.update(new_sums)
        
    return True

def _count_addable(S, candidates):
    """
    Count how many candidates can be added to S while maintaining dissociation.
    """
    count = 0
    for c in candidates:
        if c in S:
            continue
        if _is_dissociated(S + [c]):
            count += 1
    return count

def _greedy_construct(order):
    """
    Construct a dissociated set by greedily adding elements in the given order.
    """
    S = []
    for x in order:
        if _is_dissociated(S + [x]):
            S.append(x)
    return S

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    all_elements = list(range(1, 23))
    
    # Phase 1: Greedy with different orderings
    best_set = []
    
    # Try increasing order
    S_inc = _greedy_construct(all_elements)
    if len(S_inc) > len(best_set):
        best_set = S_inc
        
    # Try decreasing order
    S_dec = _greedy_construct(sorted(all_elements, reverse=True))
    if len(S_dec) > len(best_set):
        best_set = S_dec
        
    # Try prime-heavy order (primes are often good for dissociation)
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    non_primes = [x for x in all_elements if x not in primes]
    S_prime = _greedy_construct(primes + non_primes)
    if len(S_prime) > len(best_set):
        best_set = S_prime
        
    # Try random orders
    random.seed(42)
    for _ in range(20):
        order = all_elements[:]
        random.shuffle(order)
        S_rand = _greedy_construct(order)
        if len(S_rand) > len(best_set):
            best_set = S_rand
            break  # Found a good one, stop random greedy
            
    # Ensure we have a non-empty set
    if not best_set:
        best_set = [1]
        
    best_size = len(best_set)
    
    # Phase 2: Local Search
    current_set = best_set[:]
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Check if we improved
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(best_set)
            no_improve_count = 0
            
        # 1. Try to add an element
        added = False
        remaining = [x for x in all_elements if x not in current_set]
        if remaining:
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() > deadline:
                    break
                test_set = current_set + [cand]
                if _is_dissociated(test_set):
                    current_set = test_set
                    added = True
                    break
        
        if not added:
            # 2. Try swaps
            swapped = False
            in_set = set(current_set)
            remaining = [x for x in all_elements if x not in in_set]
            if remaining and current_set:
                random.shuffle(remaining)
                attempts = 0
                max_attempts = 20
                while attempts < max_attempts and time.time() < deadline:
                    attempts += 1
                    out_cand = random.choice(remaining)
                    in_cand = random.choice(current_set)
                    
                    if out_cand == in_cand:
                        continue
                    
                    new_set = [x for x in current_set if x != in_cand] + [out_cand]
                    
                    if _is_dissociated(new_set):
                        # Accept swap to explore state space
                        current_set = new_set
                        in_set = set(current_set)
                        remaining = [x for x in all_elements if x not in in_set]
                        swapped = True
                        break
            
            if not swapped:
                no_improve_count += 1
                # If stuck for a while, reset to best or random restart
                if no_improve_count > 200:
                    # Reset to best
                    current_set = best_set[:]
                    no_improve_count = 0
                    
                    # Occasionally try a random restart
                    if random.random() < 0.1:
                        k = max(1, best_size - 1)  # Try slightly smaller to allow growth
                        current_set = random.sample(all_elements, k)
                        if not _is_dissociated(current_set):
                            current_set = best_set[:]
                        
        # Time check
        if time.time() > deadline:
            break
    
    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
