import time
import random
import sys

N = 5000
DEADLINE_MARGIN = 20.0
T_START = time.time()

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _calculate_available_count(choice_map, primes_list):
    """
    choice_map: dict {p: c} where c is the missed class for prime p.
    Returns the number of integers in [0, N] that are NOT in any missed class.
    """
    # Forbidden is the union of arithmetic progressions
    # We can use a bitset or a boolean array for speed
    forbidden = bytearray(N + 1)
    
    for p in primes_list:
        c = choice_map.get(p)
        if c is None:
            continue
        # Mark all x in [0, N] such that x % p == c as forbidden
        # x = c, c+p, c+2p, ...
        for x in range(c, N + 1, p):
            forbidden[x] = 1
            
    # Count available
    # available = sum(1 for x in range(N+1) if not forbidden[x])
    # Using sum on bytearray is fast
    return (N + 1) - sum(forbidden)

def _greedy_optimize_classes(primes_list, K, initial_choice=None):
    """
    Try to find a choice of missed classes for each p in primes_list
    such that the available count is >= K.
    Uses local search / hill climbing.
    """
    if not primes_list:
        return True, {}

    # Initialize choices
    choice_map = {}
    if initial_choice:
        choice_map = dict(initial_choice)
    else:
        for p in primes_list:
            choice_map[p] = random.randint(0, p - 1)
    
    current_count = _calculate_available_count(choice_map, primes_list)
    
    if current_count >= K:
        return True, choice_map

    # Local Search: Try to improve the count by changing one class at a time
    improved = True
    iterations = 0
    max_iterations = 5000 # Safety cap per restart
    
    while improved and _time_left() > 5.0 and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Shuffle primes to avoid bias
        primes_to_try = primes_list[:]
        random.shuffle(primes_to_try)
        
        for p in primes_to_try:
            current_c = choice_map[p]
            best_c = current_c
            best_count = current_count
            
            # Try all other classes for this prime
            for c in range(p):
                if c == current_c:
                    continue
                
                # Calculate delta
                # Removing old class c_old from forbidden, adding new class c_new
                # We need to count how many elements are ONLY in c_old (not in other forbidden sets)
                # and how many are ONLY in c_new.
                
                # Efficient calculation:
                # Let F be the set of forbidden elements.
                # F = F_old \ A_{p, current_c} \cup A_{p, c}
                # We want |F_new| to be small.
                
                # Let's just recalculate the count for this specific change?
                # Too slow if done naively.
                
                # Approximation:
                # Gain = (elements in A_{p, current_c} that are not in F \ A_{p, current_c})
                #       - (elements in A_{p, c} that are not in F \ A_{p, current_c})
                
                # This requires knowing the intersection with other forbidden sets.
                # For simplicity in this step, we might just do a full recalculation
                # if the number of primes is small (< 20).
                
                if len(primes_list) < 20:
                    trial_map = dict(choice_map)
                    trial_map[p] = c
                    trial_count = _calculate_available_count(trial_map, primes_list)
                    if trial_count > best_count:
                        best_count = trial_count
                        best_c = c
                else:
                    # For larger sets, use a heuristic or skip
                    pass

            if best_c != current_c:
                choice_map[p] = best_c
                current_count = best_count
                improved = True
                
                if current_count >= K:
                    return True, choice_map

    return False, choice_map

def construct():
    # Binary search on the maximum size K
    # Lower bound: 1
    # Upper bound: N + 1
    # But K cannot be too large because we must miss classes for all p <= K.
    # The available set shrinks as K increases.
    
    # Primes up to N
    all_primes = _primes(N)
    
    best_tuple = []
    best_size = 0
    
    # We search for the largest K.
    # Start from a reasonable guess. 
    # If we miss 1 class for each p <= K, the density of available numbers is 
    # prod_{p<=K} (1 - 1/p).
    # This is approx 1 / (e^gamma * log K).
    # Available count approx N / (e^gamma * log K).
    # We need Available Count >= K.
    # N / log K >= K  =>  K log K <= N.
    # For N=5000, K ~ 500? 500 * log(500) ~ 500 * 6 = 3000 < 5000.
    # K ~ 800? 800 * 6.7 = 5360 > 5000.
    # So K is likely around 500-700.
    
    # Let's scan K from high to low.
    # But checking a specific K requires solving the class optimization.
    
    # Strategy:
    # 1. Estimate max K.
    # 2. Try to construct for K.
    # 3. If fail, decrease K.
    
    # We will try a range of K values.
    # To be safe and fast, we'll try K from 100 down to 1 if high K fails.
    # But we want the LARGEST.
    
    # Let's try K = 100, 200, ..., 1000.
    # Actually, let's just try to maximize it using a greedy construction of the tuple itself?
    # No, the class-selection strategy is better.
    
    # Let's iterate K from a high estimate downwards.
    # Estimate:
    # Let's just try K = 500 first.
    
    # We'll do a binary search on K, but the predicate "Is K feasible?" is expensive.
    # Instead, we'll do a linear search from a high bound, but with a time limit.
    
    max_K_to_try = 1000
    min_K_to_try = 1
    
    # We'll try K in descending order, but step by step.
    # To save time, we'll check K = 500, 400, 300...
    
    # Better: Use the time budget to run multiple restarts for a fixed K,
    # or adapt K.
    
    # Let's try K from 100 to 1000 in steps of 10, and keep the best valid tuple.
    # But we want the LARGEST.
    # So we should check high K first.
    
    # If K=800 is feasible, we return it.
    # If not, try 790, etc.
    
    # To avoid checking too many K values, we'll use a coarse step first.
    
    # Primes for a given K are just primes <= K.
    
    for K in range(800, 0, -10):
        if _time_left() < 30.0:
            break
            
        primes_for_K = [p for p in all_primes if p <= K]
        
        # Try to find a valid configuration for this K
        # Use multiple random restarts for the class optimization
        success = False
        best_choice_map = None
        
        restarts = 10
        for r in range(restarts):
            if _time_left() < 30.0:
                break
            feasible, choice_map = _greedy_optimize_classes(primes_for_K, K, initial_choice=None)
            if feasible:
                success = True
                best_choice_map = choice_map
                break
        
        if success:
            # Construct the tuple
            # Available set is [0, N] \ Forbidden
            forbidden = bytearray(N + 1)
            for p in primes_for_K:
                c = best_choice_map[p]
                for x in range(c, N + 1, p):
                    forbidden[x] = 1
            
            # Pick the first K available numbers
            T = []
            for x in range(N + 1):
                if not forbidden[x]:
                    T.append(x)
                    if len(T) == K:
                        break
            
            # Verify admissibility
            # For every p <= len(T) (which is K), some class is missed.
            # By construction, class c_p is missed for all p <= K.
            # So it is admissible.
            
            if len(T) == K:
                return T
            else:
                # Not enough available numbers? Should not happen if count >= K
                pass
        
        # If not successful, try smaller K
    return [1] # Fallback

# Wait, the loop `for K in range(800, 0, -10)` might miss the exact max if it's between steps.
# But the score is `value / seed`.
# If we find K=790 and the max is 795, we lose a bit.
# We can refine after finding a feasible K.

# Let's refine the search:
# 1. Find a feasible K_coarse.
# 2. Binary search between K_coarse and K_coarse + 10.

def construct():
    all_primes = _primes(N)
    best_tuple = []
    
    # Coarse search
    K_found = 0
    for K in range(800, 0, -10):
        if _time_left() < 30.0:
            break
        primes_for_K = [p for p in all_primes if p <= K]
        
        feasible = False
        best_cm = None
        for _ in range(5):
            if _time_left() < 30.0: break
            f, cm = _greedy_optimize_classes(primes_for_K, K)
            if f:
                feasible = True
                best_cm = cm
                break
        
        if feasible:
            K_found = K
            break
        else:
            # Maybe it's feasible but we didn't find the right classes?
            # We'll assume if 5 restarts fail, it's likely infeasible or very hard.
            pass
            
    if K_found == 0:
        return [1]
        
    # Fine search: K_found to K_found + 10
    best_K = K_found
    best_cm = None
    
    for K in range(K_found + 10, K_found - 1, -1):
        if K < 1: break
        if _time_left() < 10.0: break
        primes_for_K = [p for p in all_primes if p <= K]
        
        # More restarts for finer search
        for _ in range(20):
            if _time_left() < 10.0: break
            f, cm = _greedy_optimize_classes(primes_for_K, K)
            if f:
                best_K = K
                best_cm = cm
                break
        if best_K == K:
            break # Found the max in this range
            
    # Construct final tuple
    if best_cm is None:
        return [1]
        
    primes_for_final = [p for p in all_primes if p <= best_K]
    forbidden = bytearray(N + 1)
    for p in primes_for_final:
        c = best_cm[p]
        for x in range(c, N + 1, p):
            forbidden[x] = 1
            
    T = []
    for x in range(N + 1):
        if not forbidden[x]:
            T.append(x)
            if len(T) == best_K:
                break
                
    return T
