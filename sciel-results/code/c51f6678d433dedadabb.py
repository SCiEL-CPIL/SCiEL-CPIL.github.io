import time
import numpy as np
from math import gcd, isqrt
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Pre-compute prime factors for each number
    prime_factors = [[] for _ in range(MAX_N + 1)]
    for n in range(1, MAX_N + 1):
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                prime_factors[n].append(p)

    # Build conflict graph: two numbers conflict if they share a prime factor
    # For each number, store the set of other numbers it conflicts with
    conflicts = [set() for _ in range(MAX_N + 1)]
    for i in range(1, MAX_N + 1):
        for j in range(i + 1, MAX_N + 1):
            # Check if i and j share any prime factor
            common = False
            for p in prime_factors[i]:
                if p in prime_factors[j]:
                    common = True
                    break
            if common:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Branch and bound for maximum weight independent set
    # We want to select a subset S of {1,...,256} such that no two elements in S share a prime factor
    # Maximize sum(S)
    
    # Sort numbers by value descending for better pruning
    # But we need to be careful about the order
    
    # Upper bound: for each prime p, at most one number divisible by p can be selected
    # So the upper bound is the sum over all primes of the max value among remaining numbers divisible by p
    # But this is not tight because a number can be divisible by multiple primes
    
    # Better approach: use a greedy upper bound
    # For the remaining candidates, compute an upper bound on what can be added
    
    def compute_upper_bound(remaining_set):
        """Compute an upper bound on the sum achievable from remaining_set."""
        if not remaining_set:
            return 0
        
        # For each prime, find the max value in remaining_set divisible by that prime
        # Then sum these maxima (this is an overestimate since one number can cover multiple primes)
        bound = 0
        used_primes = set()
        for n in sorted(remaining_set, reverse=True):
            # Check if n conflicts with any already "counted" prime
            can_count = True
            for p in prime_factors[n]:
                if p in used_primes:
                    can_count = False
                    break
            if can_count:
                bound += n
                for p in prime_factors[n]:
                    used_primes.add(p)
        return bound

    # Branch and bound
    best_sum = 0
    best_set = set()
    
    # Start with an initial solution using greedy
    greedy_set = set()
    for n in range(MAX_N, 0, -1):
        if all(gcd(n, x) == 1 for x in greedy_set):
            greedy_set.add(n)
    best_sum = sum(greedy_set)
    best_set = set(greedy_set)
    
    # Branch and bound search
    # We'll use a recursive approach with memoization on the "state" of which numbers are still available
    # State: frozenset of available numbers
    # This might be too large, so let's use a different approach
    
    # Alternative: iterate through numbers in order, deciding to include or exclude each
    # Use branch and bound with pruning
    
    def branch_and_bound(index, current_set, current_sum, available_mask):
        """
        index: next number to consider (from MAX_N down to 1)
        current_set: set of selected numbers so far
        current_sum: sum of selected numbers
        available_mask: bitmask of which numbers are still available (bit i set means number i+1 is available)
        """
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(current_set)
        
        if index < 1:
            return
        
        # Upper bound: current_sum + sum of all remaining available numbers
        # This is a loose bound, let's use a tighter one
        remaining = []
        for n in range(index, 0, -1):
            if available_mask & (1 << n):
                remaining.append(n)
        
        if not remaining:
            return
        
        # Compute a better upper bound
        ub = current_sum
        temp_used_primes = set()
        for n in remaining:
            can_add = True
            for p in prime_factors[n]:
                if p in temp_used_primes:
                    can_add = False
                    break
            if can_add:
                ub += n
                for p in prime_factors[n]:
                    temp_used_primes.add(p)
        
        if ub <= best_sum:
            return
        
        # Try including number 'index'
        n = index
        if available_mask & (1 << n):
            # Check if n can be added (doesn't conflict with current_set)
            can_add = True
            for x in current_set:
                if x in conflicts[n]:
                    can_add = False
                    break
            
            if can_add:
                new_available = available_mask
                # Remove n and all its conflicts from available
                for c in conflicts[n]:
                    new_available &= ~(1 << c)
                new_available &= ~(1 << n)
                
                current_set.add(n)
                branch_and_bound(index - 1, current_set, current_sum + n, new_available)
                current_set.remove(n)
        
        # Try excluding number 'index'
        branch_and_bound(index - 1, current_set, current_sum, available_mask)
    
    # Initialize available mask
    initial_mask = 0
    for n in range(1, MAX_N + 1):
        initial_mask |= (1 << n)
    
    branch_and_bound(MAX_N, set(), 0, initial_mask)
    
    if best_set:
        # Verify the solution
        valid = True
        for i in range(len(list(best_set))):
            for j in range(i + 1, len(list(best_set))):
                if gcd(list(best_set)[i], list(best_set)[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [int(x) for x in sorted(best_set)]
    
    # Fallback: Local Search
    S = []
    for n in range(256, 0, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    best_set = set(S)
    best_sum = sum(S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(list(best_set))
        
        for b in range(256, 1, -1):
            if b in best_set:
                continue
            if all(gcd(b, x) == 1 for x in best_set):
                best_set.add(b)
                best_sum += b
                improved = True
        
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set.remove(a)
                        best_set.add(b)
                        best_sum = new_sum
                        improved = True
    
    if 1 not in best_set:
        if all(gcd(1, x) == 1 for x in best_set):
            best_set.add(1)
            best_sum += 1

    return [int(x) for x in sorted(best_set)]
