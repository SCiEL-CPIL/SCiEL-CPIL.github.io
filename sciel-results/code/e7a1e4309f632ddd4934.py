import math
import time
import random

def construct():
    n = 256
    deadline = time.time() + 85  # 85s safety margin
    
    # 1. Sieve and Prime Factorization
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Bitmask for prime factors
    factor_mask = [0] * (n + 1)
    factor_mask[1] = 0
    for i in range(2, n + 1):
        mask = 0
        temp = i
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_index[temp])
        factor_mask[i] = mask

    # 2. Greedy Phase with Randomization
    best_sum = 0
    best_set = []
    
    # Always include 1
    # The set of candidates is 2..256
    
    # Run a few deterministic greedy passes first
    # Pass 1: Sort by value descending
    candidates = list(range(2, n + 1))
    candidates.sort(reverse=True)
    current_set = [1]
    current_mask = 0
    current_sum = 1
    for c in candidates:
        if (current_mask & factor_mask[c]) == 0:
            current_set.append(c)
            current_mask |= factor_mask[c]
            current_sum += c
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = list(current_set)
        
    # Pass 2: Sort by value / number of prime factors
    def heuristic_val(x):
        return x / max(1, bin(factor_mask[x]).count('1'))
        
    candidates.sort(key=heuristic_val, reverse=True)
    current_set = [1]
    current_mask = 0
    current_sum = 1
    for c in candidates:
        if (current_mask & factor_mask[c]) == 0:
            current_set.append(c)
            current_mask |= factor_mask[c]
            current_sum += c
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = list(current_set)

    # Randomized Greedy
    num_restarts = 500
    for _ in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s for local search
            break
        random.shuffle(candidates)
        current_set = [1]
        current_mask = 0
        current_sum = 1
        for c in candidates:
            if (current_mask & factor_mask[c]) == 0:
                current_set.append(c)
                current_mask |= factor_mask[c]
                current_sum += c
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)

    # 3. Local Search on Best Set
    # We will try to improve best_set
    # current_set is a list of ints
    # current_mask is the OR of masks
    # current_sum is the sum
    
    # We need to be able to remove an element and check if a new one fits.
    # Since elements are pairwise coprime, removing x means the new mask is current_mask ^ factor_mask[x].
    
    improved = True
    local_search_deadline = deadline - 2
    while improved and time.time() < local_search_deadline:
        improved = False
        current_set = list(best_set)
        current_mask = 0
        for x in current_set:
            current_mask |= factor_mask[x]
        current_sum = best_sum
        
        # Try 1-for-1 swap
        # For each x in set, try replacing it with y not in set
        # Gain = y - x
        # Constraint: (current_mask ^ factor_mask[x]) & factor_mask[y] == 0
        
        # Precompute masks for all numbers 1..256
        # We already have factor_mask
        
        # To speed up, we can iterate y in descending order and check if it fits in any "slot"
        # But slots are defined by removing x.
        
        # Simpler: Iterate x, then iterate y
        # This is O(N^2) per pass. N is small (size of set is ~20-30).
        # Max set size: 1 + 54 primes = 55.
        # 55 * 256 = 14080 checks. Very fast.
        
        found_improvement = False
        best_gain = 0
        best_swap = None
        
        # 1-for-1
        for i in range(len(current_set)):
            x = current_set[i]
            mask_without_x = current_mask ^ factor_mask[x]
            # Try all y not in set
            # To be efficient, we can just check all y in 2..256
            for y in range(2, n + 1):
                if y in current_set:
                    continue
                if (mask_without_x & factor_mask[y]) == 0:
                    gain = y - x
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, y)
        
        if best_swap:
            i, y = best_swap
            current_set[i] = y
            current_mask ^= factor_mask[current_set[i]] # Wait, we need to update mask properly
            # Let's recalculate mask or update it:
            # old mask had x. new mask has y.
            # Since x and y are disjoint from others, and x and y are disjoint from each other?
            # Wait, if we swap x for y, y must be disjoint from all others.
            # The others are disjoint from x.
            # So new_mask = (current_mask ^ factor_mask[x]) | factor_mask[y]
            current_mask = (current_mask ^ factor_mask[x]) | factor_mask[y]
            current_sum += best_gain
            best_sum = current_sum
            best_set = current_set
            improved = True
            found_improvement = True
            continue # Restart loop with new state
            
        # If no 1-for-1, try 1-for-2 (Add 2, Remove 1)
        # Gain = y1 + y2 - x
        # Constraint: y1, y2 disjoint from each other and from (set \ {x})
        
        best_gain_12 = 0
        best_swap_12 = None
        
        for i in range(len(current_set)):
            if time.time() > local_search_deadline:
                break
            x = current_set[i]
            mask_without_x = current_mask ^ factor_mask[x]
            
            # Find best pair y1, y2
            # Iterate y1
            for y1 in range(2, n + 1):
                if y1 in current_set:
                    continue
                if (mask_without_x & factor_mask[y1]) != 0:
                    continue
                # y1 is compatible with rest
                # Now find best y2 compatible with rest and y1
                mask_without_x_and_y1 = mask_without_x | factor_mask[y1]
                best_y2 = -1
                for y2 in range(2, n + 1):
                    if y2 in current_set or y2 == y1:
                        continue
                    if (mask_without_x_and_y1 & factor_mask[y2]) != 0:
                        continue
                    gain = y1 + y2 - x
                    if gain > best_gain_12:
                        best_gain_12 = gain
                        best_swap_12 = (i, y1, y2)
        
        if best_swap_12 and best_gain_12 > 0:
            i, y1, y2 = best_swap_12
            x = current_set[i]
            # Remove x, add y1, y2
            current_set.pop(i)
            current_set.append(y1)
            current_set.append(y2)
            # Update mask
            current_mask = (current_mask ^ factor_mask[x]) | factor_mask[y1] | factor_mask[y2]
            current_sum += best_gain_12
            best_sum = current_sum
            best_set = current_set
            improved = True
            found_improvement = True
            continue # Restart
            
        # If no 1-for-2, try 2-for-1 (Add 1, Remove 2)
        # Gain = y - x1 - x2
        # Constraint: y disjoint from (set \ {x1, x2})
        
        best_gain_21 = 0
        best_swap_21 = None
        
        for i in range(len(current_set)):
            if time.time() > local_search_deadline:
                break
            for j in range(i + 1, len(current_set)):
                x1 = current_set[i]
                x2 = current_set[j]
                mask_without_x1_x2 = current_mask ^ factor_mask[x1] ^ factor_mask[x2]
                
                # Find best y
                for y in range(2, n + 1):
                    if y in current_set:
                        continue
                    if (mask_without_x1_x2 & factor_mask[y]) == 0:
                        gain = y - x1 - x2
                        if gain > best_gain_21:
                            best_gain_21 = gain
                            best_swap_21 = (i, j, y)
        
        if best_swap_21 and best_gain_21 > 0:
            i, j, y = best_swap_21
            x1 = current_set[i]
            x2 = current_set[j]
            # Remove x1, x2, add y
            # Order matters for indices. Remove higher index first.
            if i > j:
                i, j = j, i
            current_set.pop(i)
            current_set.pop(j)
            current_set.append(y)
            current_mask = (current_mask ^ factor_mask[x1] ^ factor_mask[x2]) | factor_mask[y]
            current_sum += best_gain_21
            best_sum = current_sum
            best_set = current_set
            improved = True
            found_improvement = True
            continue # Restart
            
        # If no improvement found in any move type, we are at a local optimum.
        # To escape, we can do a random perturbation (Simulated Annealing style)
        # Or just stop if time is running out.
        # Given the time budget, if we are stuck, we might try a random restart of local search
        # from a slightly perturbed state, but that's risky.
        # Let's just break if no improvement.
        
    return [int(x) for x in best_set]
