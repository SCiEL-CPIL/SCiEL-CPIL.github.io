import time
import random

def construct():
    n = 256
    # Initial best solution: interval [171, 256]
    # Size 86
    best_set = list(range(171, n + 1))
    best_size = len(best_set)
    
    def is_valid(S):
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Verify initial solution
    if not is_valid(best_set):
        # Fallback to safe upper tail
        best_set = list(range(192, n + 1))
        best_size = len(best_set)

    # Try to improve with local search (swaps)
    # Time limit: 60s
    start_time = time.time()
    time_limit = 55.0
    
    # Generate candidates for addition (smaller numbers) and removal (larger numbers)
    # We try to replace a subset of large numbers with a subset of small numbers
    # Or just single swaps: remove x, add y.
    
    # To make it faster, we only check validity of the changed parts?
    # No, full check is O(N^3) which is ~ 256^3 = 16M ops. 
    # Python might be slow. 16M ops in 55s is ~ 300 ops/ms.
    # 16M / 0.3M/s = 50s. One check takes too long.
    # We need a faster validity check or fewer checks.
    
    # Optimized check:
    # For a given set S, precompute sums?
    # Or just check pairs involving the changed element.
    
    def check_addition(S, new_elem):
        # Check if adding new_elem to S keeps it valid
        # Only need to check triples involving new_elem
        # 1. new_elem is a: new_elem < b <= c. new_elem must not divide b+c
        # 2. new_elem is b: a < new_elem <= c. a must not divide new_elem+c
        # 3. new_elem is c: a < b <= new_elem. a must not divide b+new_elem
        
        S_sorted = sorted(S)
        
        # Case 1: a = new_elem
        # b, c in S, b > new_elem, b <= c
        # b+c must not be divisible by new_elem
        for b in S_sorted:
            if b <= new_elem:
                continue
            # c >= b
            for c in S_sorted:
                if c < b:
                    continue
                if (b + c) % new_elem == 0:
                    return False
        
        # Case 2: b = new_elem
        # a in S, a < new_elem. c in S, c >= new_elem
        # a must not divide new_elem + c
        for a in S_sorted:
            if a >= new_elem:
                continue
            for c in S_sorted:
                if c < new_elem:
                    continue
                if (new_elem + c) % a == 0:
                    return False
        
        # Case 3: c = new_elem
        # a in S, a < b in S. b <= new_elem
        # a must not divide b + new_elem
        for a in S_sorted:
            for b in S_sorted:
                if a >= b:
                    continue
                if b > new_elem:
                    continue
                if (b + new_elem) % a == 0:
                    return False
                    
        return True

    def check_removal(S, rem_elem):
        # Removing an element always keeps the set valid if it was valid before
        # because we are removing constraints.
        return True

    # Local Search: Try to add a small number and remove a large number
    # If we add x and remove y, size stays same.
    # If we add x and don't remove, size increases.
    # We first try to just add numbers (greedy).
    
    current_set = list(best_set)
    
    # Try adding numbers from 1 to 170
    for candidate in range(1, 171):
        if time.time() - start_time > time_limit:
            break
        if candidate not in current_set:
            if check_addition(current_set, candidate):
                current_set.append(candidate)
                best_size += 1
    
    # Now try swaps: remove y from current_set, add x not in current_set
    # We prioritize adding smaller x and removing larger y
    # To avoid O(N^2) full checks, we can be smart.
    
    # Let's try a few random restarts or just systematic swaps
    # Given time, systematic is better.
    
    # List of elements in current_set
    # List of elements NOT in current_set
    in_set = set(current_set)
    not_in_set = [i for i in range(1, n+1) if i not in in_set]
    
    # Try to improve by swapping
    # We iterate through candidates to add (small to large)
    # and candidates to remove (large to small)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Sort not_in_set ascending, current_set descending
        # Try to add small x
        for x in not_in_set:
            if time.time() - start_time > time_limit:
                break
            if x in in_set: continue
            # Try to remove y
            # If we can add x without removing, we would have done it in the greedy phase.
            # So we MUST remove at least one y.
            # Try removing one y
            for y in sorted(current_set, reverse=True):
                if y == x: continue
                # Check if removing y and adding x is valid
                # New set: current_set - {y} + {x}
                # We need to check validity of the new set.
                # Fast check:
                # 1. Check if current_set - {y} is valid (always true)
                # 2. Check if adding x to current_set - {y} is valid
                temp_set = [v for v in current_set if v != y]
                if check_addition(temp_set, x):
                    # Perform swap
                    current_set = temp_set
                    current_set.append(x)
                    in_set.remove(y)
                    in_set.add(x)
                    # Update not_in_set
                    not_in_set.remove(x)
                    not_in_set.append(y)
                    improved = True
                    break
            if improved:
                break # Restart loop with new set
                
    return current_set
