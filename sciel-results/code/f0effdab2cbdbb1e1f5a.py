import time
import random
from math import gcd

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # Initial candidate: Divisors of 480
    # 480 = 2^5 * 3 * 5, has 24 divisors
    base_num = 480
    current_set = set()
    for i in range(1, base_num + 1):
        if base_num % i == 0:
            current_set.add(i)
    
    # Also consider other highly composite numbers <= 512
    # 420, 360 also have 24 divisors.
    # We'll stick with 480 as it's a multiple of many small numbers.
    
    best_size = len(current_set)
    best_set = set(current_set)
    
    def is_compatible(x, s):
        for y in s:
            if lcm(x, y) > 512:
                return False
        return True
    
    # Greedy addition
    # Try to add any number from 1 to 512 not in set
    for x in range(1, 513):
        if x in current_set:
            continue
        if time.time() > deadline:
            break
        if is_compatible(x, current_set):
            current_set.add(x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
    
    # Local Search: Swap
    # Try removing one element and adding one or two others
    # This is more complex, so we do a simpler version:
    # Try to replace a "low utility" element with a new one.
    # Utility is hard to define simply.
    # Instead, try all single swaps: remove a, add x.
    # If we can add x without removing, we did that above.
    # So only consider swaps where x is NOT compatible with current_set.
    
    # We will iterate a few times
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 50:
        improved = False
        iterations += 1
        
        # Create a list of elements to try removing
        # Shuffle for randomness? Or just iterate.
        elements = list(current_set)
        
        for a in elements:
            if time.time() > deadline:
                break
            
            # Temporarily remove a
            temp_set = set(current_set)
            temp_set.remove(a)
            
            # Try to add any x not in temp_set
            added_any = False
            # We want to maximize size, so adding 1 is enough to maintain size,
            # but we need to find if we can add MORE than we removed.
            # Since we removed 1, we need to add 2 to increase size.
            # Or add 1 to maintain, and maybe find a better configuration?
            # Actually, if we just add 1, size stays same.
            # We are looking for size increase.
            # So we need to find x1, x2 such that both are compatible with temp_set
            # and compatible with each other, and not in temp_set.
            
            # This is a sub-problem: Max Clique in the subgraph induced by
            # candidates compatible with temp_set.
            # Given small size, we can just try pairs.
            
            # First, find all candidates compatible with temp_set
            candidates = []
            for x in range(1, 513):
                if x in temp_set:
                    continue
                if is_compatible(x, temp_set):
                    candidates.append(x)
            
            # Now find a pair (x1, x2) in candidates such that lcm(x1, x2) <= 512
            # And we want to add as many as possible?
            # If we add 2, size increases by 1.
            # If we add 3, size increases by 2.
            
            # Greedy pick from candidates
            # Sort candidates?
            # Just try to find a clique of size 2 in candidates.
            found_pair = False
            # To speed up, limit candidates check
            # candidates list can be large.
            # But we only need to find ONE pair to improve size by 1.
            
            # Optimization:
            # If len(candidates) < 2, no improvement possible with this removal.
            if len(candidates) >= 2:
                # Check pairs
                # This is O(M^2) where M is len(candidates)
                # M can be up to 512. 512^2 is 260k.
                # Doing this for each of 24 removals is 6M ops.
                # Might be slow if done many times.
                # But we break early if found.
                
                for i in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    x1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        if time.time() > deadline:
                            break
                        x2 = candidates[j]
                        if lcm(x1, x2) <= 512:
                            # We found a pair
                            # Add them to temp_set
                            temp_set.add(x1)
                            temp_set.add(x2)
                            current_set = temp_set
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                # If we found a pair, we improved.
                # We break out of the 'for a' loop to restart search with new set?
                # Or continue?
                # Usually, restart is safer to avoid local minima traps.
                if found_pair:
                    break

    # Final check: maybe we can add more to the best_set found?
    # The best_set might be stale if current_set improved.
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
        
    # One final greedy pass on best_set to ensure no single additions missed
    # (The swap logic might have changed the set, making previously incompatible items compatible?
    # No, removing items only makes it EASIER to add.
    # So if we found a swap, the new set is larger.
    # But maybe we can add MORE to the new set?
    # The greedy addition at the start was on the OLD set.
    # So we should run greedy addition on the FINAL best_set.
    
    final_set = set(best_set)
    for x in range(1, 513):
        if x in final_set:
            continue
        if is_compatible(x, final_set):
            final_set.add(x)
            
    if len(final_set) > best_size:
        best_set = final_set
        best_size = len(final_set)

    return list(best_set)
