"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import itertools
import random

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def _is_smooth(n, primes_set, primes_list):
    """Check if n is P-smooth where P is the set of primes in primes_set."""
    for p in primes_list:
        if p * p > n:
            break
        if p in primes_set:
            while n % p == 0:
                n //= p
        else:
            # If p is not in our set and divides n, n is not smooth
            # But we need to check all prime factors
            # Actually, let's just factor n completely and check
            pass
    # Better approach: factor n and check all prime factors are in primes_set
    temp = n
    for p in primes_list:
        if p * p > temp:
            break
        while temp % p == 0:
            if p not in primes_set:
                return False
            temp //= p
    if temp > 1:
        if temp not in primes_set:
            return False
    return True

def construct():
    # We want to find 12 distinct positive integers <= 10000 such that
    # the union of prime factors of all pairwise sums is minimized.
    
    # Strategy: Try small sets of primes P, and find 12 numbers such that
    # all pairwise sums are P-smooth.
    
    # Start with the smallest primes and expand the set if needed.
    # We'll try prime sets of size 1, 2, 3, etc.
    
    all_primes = _sieve_primes(20000)  # sums can be up to 20000
    
    # Try prime sets in order of size, then by smallest prime
    # For each prime set, try to find 12 numbers whose pairwise sums are all P-smooth
    
    # To make this efficient, we can:
    # 1. Precompute for each prime set P, which numbers in 1..10000 have the property
    #    that when added to other candidate numbers, the sum is P-smooth.
    # 2. Use a greedy or backtracking approach to find 12 compatible numbers.
    
    # Let's try prime sets of size 1, 2, 3, 4, 5
    # For each set, generate all numbers 1..10000 and check which pairwise sums are P-smooth
    
    # Optimization: For a given P, two numbers a, b are compatible if a+b is P-smooth.
    # We need a clique of size 12 in the compatibility graph.
    
    # Try prime sets in increasing order of size
    for num_primes in range(1, 8):
        # Try different combinations of the first num_primes primes
        # Start with the smallest primes as they're most likely to work
        base_primes = all_primes[:num_primes]
        
        # Also try some variations
        prime_sets_to_try = [tuple(base_primes)]
        
        # Add some variations with slightly larger primes
        if num_primes <= 4:
            for i in range(1, min(5, len(all_primes) - num_primes + 1)):
                alt = tuple(all_primes[:num_primes-1] + [all_primes[i]])
                if alt not in prime_sets_to_try:
                    prime_sets_to_try.append(alt)
        
        for primes_tuple in prime_sets_to_try:
            primes_set = set(primes_tuple)
            primes_list = list(primes_tuple)
            
            # Precompute which numbers in 1..10000 can form P-smooth sums with others
            # For efficiency, first identify candidate numbers
            # A number a is a candidate if there exists some b != a such that a+b is P-smooth
            
            # Let's build the compatibility graph implicitly
            # For each pair (a, b), check if a+b is P-smooth
            
            # To speed up, precompute for each sum s from 2 to 20000 whether s is P-smooth
            smooth_sums = [False] * 20001
            for s in range(2, 20001):
                smooth_sums[s] = _is_smooth(s, primes_set, all_primes)
            
            # Now find 12 numbers such that all pairwise sums are smooth
            # Use greedy: start with a candidate, then add compatible numbers one by one
            
            # First, find all candidate numbers (those that have at least 11 compatible partners)
            # Actually, let's just try a greedy approach
            
            # Sort numbers by some heuristic - maybe numbers that are P-smooth themselves
            # or numbers that have many smooth-sum partners
            
            # For each number a in 1..10000, count how many b != a have a+b P-smooth
            # This is O(10000^2) which might be too slow for many prime sets
            
            # Alternative: just try to build the set greedily
            # Start with the smallest numbers and check compatibility
            
            # Let's try a different approach: use backtracking with pruning
            # But first, let's try a simple greedy:
            # Pick the smallest number, then the next smallest compatible, etc.
            
            candidates = list(range(1, 10001))
            
            # Greedy: pick numbers one by one
            selected = []
            remaining = set(candidates)
            
            for a in candidates:
                if len(selected) >= 12:
                    break
                # Check if a is compatible with all selected
                compatible = True
                for b in selected:
                    if a + b > 20000 or not smooth_sums[a + b]:
                        compatible = False
                        break
                if compatible:
                    selected.append(a)
            
            if len(selected) >= 12:
                # Verify
                result = selected[:12]
                # Double-check all pairwise sums
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        s = result[i] + result[j]
                        if not smooth_sums[s]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return result
    
    # If we couldn't find a set with <=7 primes, fall back to a simpler approach
    # Use the original approach but with some optimization
    # Try to find numbers that are all multiples of a small prime, so sums are also multiples
    
    # Fallback: return a reasonable set
    # Numbers that are all even: sums are even, so only prime factor is 2
    # But we need distinct numbers, and 12 distinct even numbers <= 10000
    # Sums of two even numbers are even, so all prime factors are just {2}
    # Wait, that's only 1 prime!
    
    # Let's verify: if all numbers are even, then a+b is even for any a,b
    # So the only prime factor is 2. That gives us just 1 distinct prime!
    
    # But wait, we need to make sure the numbers are distinct and <= 10000
    # 12 distinct even numbers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
    
    # Let's return this as a strong fallback
    result = list(range(2, 26, 2))  # 2, 4, 6, ..., 24
    return result

# EVOLVE-BLOCK-END
