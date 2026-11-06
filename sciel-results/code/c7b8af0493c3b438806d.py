import time
import heapq
import math

def construct():
    deadline = time.time() + 50.0
    N_MAX = 1000000
    N_MIN = 10000
    
    # First, compute LPF for all numbers up to N_MAX+1 using the sieve
    # This is needed to evaluate candidates quickly
    limit = N_MAX + 1
    lpf = [0] * (limit + 1)
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
        if time.time() > deadline:
            break
    
    # Find initial best by scanning
    best_score = max(lpf[N_MIN], lpf[N_MIN + 1])
    best_n = N_MIN
    
    for n in range(N_MIN + 1, N_MAX + 1):
        current_score = max(lpf[n], lpf[n + 1])
        if current_score < best_score:
            best_score = current_score
            best_n = n
        if time.time() > deadline:
            break
    
    # Now use a priority queue to explore more efficiently
    # Generate numbers in order of increasing LPF
    # Start with small primes and build up
    
    # We'll use a different strategy: 
    # For each possible max prime factor P, check if there exist consecutive
    # numbers n, n+1 in [N_MIN, N_MAX] both having all prime factors <= P
    
    # Let's try a smarter approach:
    # Precompute for each number its LPF, then for each candidate threshold T,
    # check if any consecutive pair both have LPF <= T
    
    # But that's O(N) per threshold. Instead, let's use the observation that
    # we want to find pairs where max(lpf[n], lpf[n+1]) is minimized.
    
    # Alternative: Use a heap to generate numbers by their LPF value
    # Start with all primes, then multiply by smaller primes
    
    # Actually, let's try a direct approach:
    # For each n from N_MIN to N_MAX, we already have lpf[n] and lpf[n+1]
    # The current approach scans all of them. Let's see if we can do better.
    
    # Key insight: If we sort numbers by their LPF, we can stop early
    # when we find two consecutive numbers both below our current best.
    
    # Let's create a list of (lpf_value, number) for all numbers in range
    # and sort by lpf_value
    if time.time() > deadline - 5:
        return best_n
    
    # Create list of (lpf_value, number) for numbers in [N_MIN, N_MAX+1]
    # We need both n and n+1, so consider numbers from N_MIN to N_MAX+1
    candidates = []
    for n in range(N_MIN, N_MAX + 2):
        candidates.append((lpf[n], n))
    
    # Sort by LPF value
    candidates.sort()
    
    # Now scan through in order of increasing LPF
    # Keep track of which numbers we've seen with LPF <= current threshold
    seen = set()
    best_found = best_score
    
    for lpf_val, num in candidates:
        if lpf_val > best_found:
            break
        
        # Check if num-1 or num+1 is also in seen (consecutive)
        if num - 1 in seen and num - 1 >= N_MIN and num - 1 <= N_MAX:
            # Found consecutive pair (num-1, num) both with LPF <= lpf_val
            pair_max = max(lpf[num - 1], lpf[num])
            if pair_max < best_found:
                best_found = pair_max
                best_n = num - 1
        if num + 1 in seen and num + 1 >= N_MIN and num + 1 <= N_MAX:
            # Found consecutive pair (num, num+1) both with LPF <= lpf_val
            pair_max = max(lpf[num], lpf[num + 1])
            if pair_max < best_found:
                best_found = pair_max
                best_n = num
        
        seen.add(num)
        
        if time.time() > deadline:
            break
    
    return best_n
