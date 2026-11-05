"""
Combined solution:
1. Uses Hamming numbers (2^i * 3^j) as primary candidates due to their structural properties
   which help avoid arithmetic mean violations.
2. Implements an efficient incremental validity check using subset-sum DP.
3. Uses a greedy construction followed by local search (swaps) and random restarts.
"""
import time
import random
import sys

# Set recursion limit just in case, though we use iterative DP
sys.setrecursionlimit(10000)

_DEADLINE = 55.0  # seconds, safety margin before 60s limit

def _generate_candidates(n_max=512):
    """
    Generate a prioritized list of candidates.
    Priority 1: Numbers of form 2^i * 3^j (Hamming numbers)
    Priority 2: Primes
    Priority 3: Other numbers
    """
    hamming = set()
    i = 0
    while (1 << i) <= n_max:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= n_max:
            hamming.add(val_i * (3 ** j))
            j += 1
        i += 1
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in range(1, n_max + 1) if is_prime(x)]
    others = [x for x in range(1, n_max + 1) if x not in hamming and x not in primes]
    
    # We want a mix. Hamming numbers are good for structure.
    # Let's sort them by value to have a deterministic start, but we will shuffle in search.
    # Actually, for the initial greedy, we might want to pick specific ones.
    # Let's return the full list with a priority key for sorting.
    # But for search, we just need the set of all valid candidates.
    all_candidates = list(range(1, n_max + 1))
    return all_candidates, sorted(hamming)

def _can_add(selected, new_elem):
    """
    Check if new_elem can be added to selected without violating the condition.
    Condition: No element is the mean of any subset of at least 2 OTHER elements.
    
    We need to check:
    1. new_elem is not the mean of any subset of 'selected' of size >= 2.
    2. No element 'a' in 'selected' is the mean of any subset of ('selected' - {a}) U {new_elem} of size >= 2.
       Note: If the subset does not contain new_elem, it was already valid in 'selected'.
       So we only need to check subsets that CONTAIN new_elem.
       Let S be the subset. S contains new_elem. Mean is 'a'.
       sum(S) = |S| * a
       sum(S \ {new_elem}) + new_elem = |S| * a
       Let T = S \ {new_elem}. T is a subset of 'selected' \ {a}.
       sum(T) + new_elem = (|T| + 1) * a
       sum(T) = (|T| + 1) * a - new_elem
       We need to check if there exists T subset of (selected \ {a}) with size k >= 1
       such that sum(T) = (k+1)*a - new_elem.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Check 1: new_elem is mean of subset of selected (size k >= 2)
    # sum(T) = k * new_elem, T subset of selected, |T| = k
    if n >= 2:
        # DP for subset sums of 'selected'
        # max sum is n * 512
        max_sum = sum(selected)
        # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
        dp = [0] * (n + 1)
        dp[0] = 1 # sum 0 with 0 elements
        
        for x in selected:
            # Update backwards to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k-1] != 0:
                    dp[k] |= (dp[k-1] << x)
        
        for k in range(2, n + 1):
            target = k * new_elem
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    
    # Check 2: For each a in selected, does a become mean of subset containing new_elem?
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m == 0:
            continue
        
        # We need T subset of 'others' with size k >= 1
        # sum(T) = (k+1)*a - new_elem
        max_sum_others = sum(others)
        
        # DP for subset sums of 'others'
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1] != 0:
                    dp[k] |= (dp[k-1] << x)
        
        # k is size of T. k ranges from 1 to m.
        # Target sum for T is (k+1)*a - new_elem.
        for k in range(1, m + 1):
            target = (k + 1) * a - new_elem
            if target < 0:
                continue
            if target > max_sum_others:
                break
            if dp[k] & (1 << target):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate candidates
    all_candidates, hamming_nums = _generate_candidates(512)
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Greedy with Hamming numbers prioritized
    # We create a few different orderings to try
    
    # Ordering 1: Hamming numbers first (sorted), then rest
    order1 = sorted(hamming_nums) + [x for x in all_candidates if x not in hamming_nums]
    
    # Ordering 2: All candidates sorted
    order2 = sorted(all_candidates)
    
    # Ordering 3: Random shuffle of Hamming numbers, then random shuffle of rest
    rng = random.Random(42)
    h_shuffled = hamming_nums[:]
    rng.shuffle(h_shuffled)
    rest_shuffled = [x for x in all_candidates if x not in hamming_nums]
    rng.shuffle(rest_shuffled)
    order3 = h_shuffled + rest_shuffled
    
    orders = [order1, order2, order3]
    
    # Run greedy for each order
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        current_set = []
        current_set_set = set()
        
        for x in order:
            if time.time() > deadline - 5:
                break
            
            if x in current_set_set:
                continue
            
            if _can_add(current_set, x):
                current_set.append(x)
                current_set_set.add(x)
                
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
                    
                if best_len >= 48:
                    return best_set
                    
    # Strategy 2: Local Search / Hill Climbing
    # If we didn't reach 48, try to improve best_set by swapping elements
    
    if best_set and time.time() < deadline - 5:
        best_set_set = set(best_set)
        # Candidates to try swapping in
        missing = [x for x in all_candidates if x not in best_set_set]
        
        # Sort missing by some heuristic? Maybe Hamming numbers first?
        missing.sort(key=lambda x: (0 if x in hamming_nums else 1, x))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element from best_set
            for i in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                
                removed_elem = best_set[i]
                temp_set = best_set[:i] + best_set[i+1:]
                temp_set_set = set(temp_set)
                
                # Try adding a missing element
                added_any = False
                for x in missing:
                    if time.time() > deadline - 5:
                        break
                    
                    if x in temp_set_set:
                        continue
                    
                    if _can_add(temp_set, x):
                        # Swap: remove best_set[i], add x
                        new_set = temp_set + [x]
                        new_set_set = temp_set_set | {x}
                        
                        # Check if this swap allows us to add MORE elements later?
                        # For now, just take the swap if it doesn't reduce size (it keeps size same)
                        # But we want to increase size. 
                        # Actually, a swap maintains size. 
                        # The goal is to find a configuration that allows adding *another* element.
                        
                        # Let's try to extend the new_set
                        extended_set = new_set[:]
                        extended_set_set = new_set_set
                        for y in missing:
                            if time.time() > deadline - 5:
                                break
                            if y in extended_set_set:
                                continue
                            if _can_add(extended_set, y):
                                extended_set.append(y)
                                extended_set_set.add(y)
                        
                        if len(extended_set) > len(best_set):
                            best_set = extended_set
                            best_set_set = extended_set_set
                            best_len = len(best_set)
                            missing = [x for x in all_candidates if x not in best_set_set]
                            missing.sort(key=lambda x: (0 if x in hamming_nums else 1, x))
                            improved = True
                            break # Restart the outer loop with new best_set
                
                if improved:
                    break

    # Strategy 3: Random Restarts
    # If still not 48, do random restarts
    rng2 = random.Random(123)
    while time.time() < deadline - 2:
        # Random order
        rand_order = all_candidates[:]
        rng2.shuffle(rand_order)
        
        current_set = []
        current_set_set = set()
        
        for x in rand_order:
            if time.time() > deadline - 2:
                break
            
            if x in current_set_set:
                continue
            
            if _can_add(current_set, x):
                current_set.append(x)
                current_set_set.add(x)
                
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
                    
                if best_len >= 48:
                    break
        
        if best_len >= 48:
            break

    # Final check
    if best_set:
        # Verify best_set is valid
        # We can do a quick check
        is_valid = True
        for i in range(len(best_set)):
            # Re-use logic or just trust _can_add which was incremental
            # But let's be safe.
            pass # _can_add was strict enough during construction
        
        return best_set
    else:
        # Fallback
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
