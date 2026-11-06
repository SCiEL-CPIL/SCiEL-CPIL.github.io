"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Structural approach: bounded search over prime exponent vectors.
We explore:
  - Square-free numbers with 14-15 distinct primes (varying which primes)
  - Numbers with 13-14 primes where one or two primes have exponent 2
We compute tau_perp for each candidate and return the best.
"""
import math
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    # Get all divisors
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j+1]) == 1:
            count += 1
    return count

def _tau_perp_from_exponents(primes, exponents):
    """Compute tau_perp more efficiently by generating divisors from factorization."""
    # Generate all divisors from prime factorization
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    divisors.sort()
    
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j+1]) == 1:
            count += 1
    return count

def _estimate_coprime_density(primes, exponents):
    """
    Estimate the density of coprime adjacencies using a greedy interval-merging heuristic.
    This models the sorted divisor lattice as a sequence of intervals where adjacent
    coprime pairs correspond to transitions between disjoint prime supports.
    
    The key insight: for a number with prime factorization p1^e1 * p2^e2 * ... * pk^ek,
    the divisors can be grouped by their "support" (which primes divide them). Adjacent
    divisors in sorted order are coprime if and only if their supports are disjoint.
    
    We estimate the number of support-disjoint boundaries by considering how the
    divisors are distributed in log-space and how often consecutive divisors have
    disjoint prime supports.
    """
    if not primes:
        return 0
    
    # For each divisor, compute its "support mask" (which primes divide it)
    # and its value. Then sort by value and count adjacent pairs with disjoint supports.
    
    # Generate divisors with their support masks
    divisors = [(1, 0)]  # (value, support_mask)
    for i, (p, e) in enumerate(zip(primes, exponents)):
        new_divs = []
        for val, mask in divisors:
            for k in range(e + 1):
                if k == 0:
                    new_divs.append((val, mask))
                else:
                    new_divs.append((val * (p ** k), mask | (1 << i)))
        divisors = new_divs
    
    # Sort by value
    divisors.sort(key=lambda x: x[0])
    
    # Count adjacent pairs with disjoint supports
    count = 0
    for j in range(len(divisors) - 1):
        if (divisors[j][1] & divisors[j+1][1]) == 0:
            count += 1
    
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Base primes (first 25 primes)
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 1
    best_score = 0
    
    # Phase 1: Square-free with 15 primes (the primorial)
    # Try the standard first-15-primes
    p15 = base_primes[:15]
    n15 = 1
    for p in p15:
        n15 *= p
    if n15 <= 10**18:
        score = _tau_perp_from_exponents(p15, [1]*15)
        if score > best_score:
            best_score = score
            best_n = n15
    
    # Phase 2: Greedy prime selection using density estimation
    # Start with the smallest primes and greedily add/swap primes to maximize density
    if time.time() > deadline:
        return best_n
    
    # Try different starting sets and greedily optimize
    for num_primes in range(13, 16):
        if time.time() > deadline:
            break
        
        # Start with the first num_primes primes
        current_primes = base_primes[:num_primes]
        current_product = 1
        for p in current_primes:
            current_product *= p
        
        if current_product > 10**18:
            continue
        
        # Greedily try to improve by swapping the largest prime with a larger one
        # or by adding another prime if we have room
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try swapping the largest prime with a larger prime
            if len(current_primes) >= 2:
                largest_idx = len(current_primes) - 1
                largest_prime = current_primes[largest_idx]
                
                # Try replacing with larger primes
                for new_prime in base_primes[num_primes:]:
                    if new_prime <= largest_prime:
                        continue
                    
                    new_product = current_product // largest_prime * new_prime
                    if new_product > 10**18:
                        break
                    
                    # Estimate the score
                    new_primes = current_primes[:largest_idx] + [new_prime]
                    est_score = _estimate_coprime_density(new_primes, [1]*num_primes)
                    
                    # Only compute exact score if estimate is promising
                    if est_score >= best_score - 2:
                        exact_score = _tau_perp_from_exponents(new_primes, [1]*num_primes)
                        if exact_score > best_score:
                            best_score = exact_score
                            best_n = new_product
                            improved = True
                        if exact_score > _estimate_coprime_density(current_primes, [1]*num_primes):
                            current_primes = new_primes
                            current_product = new_product
                            improved = True
            
            # Try adding another prime if we have room
            if len(current_primes) < 15:
                for new_prime in base_primes[len(current_primes):]:
                    new_product = current_product * new_prime
                    if new_product > 10**18:
                        break
                    
                    new_primes = current_primes + [new_prime]
                    est_score = _estimate_coprime_density(new_primes, [1]*(len(current_primes)+1))
                    
                    if est_score >= best_score - 2:
                        exact_score = _tau_perp_from_exponents(new_primes, [1]*(len(current_primes)+1))
                        if exact_score > best_score:
                            best_score = exact_score
                            best_n = new_product
                        current_primes = new_primes
                        current_product = new_product
                        improved = True
                        break  # Only add one prime per iteration
    
    # Phase 3: Try 14 primes with one prime squared
    # n = p1^2 * p2 * p3 * ... * p14
    # Product must be <= 10^18
    if time.time() > deadline:
        return best_n
    
    # Try squaring the smallest primes (2, 3, 5) since they have the least impact on product
    for square_prime_idx in range(4):  # Try squaring primes at index 0,1,2,3
        if time.time() > deadline:
            break
        sq_prime = base_primes[square_prime_idx]
        # Use 14 primes total: the squared prime + 13 others
        # The product is sq_prime^2 * (product of 13 other primes)
        # We want to maximize the number of divisors and coprime adjacencies
        
        # Try using the first 14 primes with one squared
        for other_start in range(0, 5):
            if time.time() > deadline:
                break
            # Choose 13 primes from base_primes, excluding the squared one if it's in range
            # Simple approach: use consecutive primes
            candidate_primes = []
            prod = sq_prime ** 2
            # Add 13 more distinct primes
            for i in range(15):
                if i == square_prime_idx:
                    continue
                p = base_primes[i]
                if prod * p > 10**18:
                    break
                candidate_primes.append(p)
                prod *= p
                if len(candidate_primes) == 13:
                    break
            
            if len(candidate_primes) == 13:
                primes_list = [sq_prime] + candidate_primes
                exponents = [2] + [1]*13
                n = prod
                if n <= 10**18:
                    score = _tau_perp_from_exponents(primes_list, exponents)
                    if score > best_score:
                        best_score = score
                        best_n = n
    
    # Phase 4: Try 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1_idx, sq2_idx in itertools.combinations(range(5), 2):
        if time.time() > deadline:
            break
        sq1 = base_primes[sq1_idx]
        sq2 = base_primes[sq2_idx]
        prod = sq1 ** 2 * sq2 ** 2
        
        # Add 11 more distinct primes
        candidate_primes = []
        for i in range(15):
            if i == sq1_idx or i == sq2_idx:
                continue
            p = base_primes[i]
            if prod * p > 10**18:
                break
            candidate_primes.append(p)
            prod *= p
            if len(candidate_primes) == 11:
                break
        
        if len(candidate_primes) == 11:
            primes_list = [sq1, sq2] + candidate_primes
            exponents = [2, 2] + [1]*11
            n = prod
            if n <= 10**18:
                score = _tau_perp_from_exponents(primes_list, exponents)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
