"""Combined approach: structured search + prime-sumset construction."""
import time
from math import gcd
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin under 90s wall-clock
    random.seed(42)
    
    # Precompute primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(256)
    prime_set = set(primes)
    
    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        if len(distinct) < 2:
            return True
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Return (valid, cardinality) for a pair."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if len(distinct) < 2:
            return (False, 0)
        if not is_pairwise_coprime(sums):
            return (False, 0)
        return (True, len(distinct))
    
    # Start with known good solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # |{5,11,17,23,29}| = 5
    
    # Strategy 1: Direct search for 9 distinct prime sums
    # If we can find A, B such that all 9 sums a_i + b_j are distinct primes,
    # we achieve maximum cardinality 9 with guaranteed pairwise coprimality.
    
    def check_prime_sumset(A, B):
        """Check if all 9 sums are distinct primes."""
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct = set(sums)
        if len(distinct) != 9:
            return False
        return all(s in prime_set for s in distinct)
    
    # Systematic search for prime sumsets
    # Key insight: for sums to be prime, parity matters.
    # If A has mixed parity and B has mixed parity, we get mixed parity sums.
    # But primes > 2 are odd, so we need most sums to be odd.
    
    # Let's try: A = {1, a, b} where a, b are even (so 1+a, 1+b are odd)
    # B = {c, d, e} where c, d, e are even (so 1+c, 1+d, 1+e are odd)
    # Then a+c, a+d, a+e, b+c, b+d, b+e are all even (not prime unless = 2)
    # This doesn't work.
    
    # Alternative: A = {1, a, b} where a, b are odd
    # B = {c, d, e} where c, d, e are odd
    # Then 1+c, 1+d, 1+e are even (not prime unless = 2)
    # a+c, a+d, a+e, b+c, b+d, b+e are even (not prime unless = 2)
    # This doesn't work either.
    
    # Better: A = {1, a, b} where a is even, b is odd
    # B = {c, d, e} where c is even, d is odd, e is odd
    # Sums:
    # 1+c (odd), 1+d (even), 1+e (even)
    # a+c (even), a+d (odd), a+e (odd)
    # b+c (odd), b+d (even), b+e (even)
    # Only 4 odd sums, 5 even sums. Even sums can only be prime if = 2.
    # This is too restrictive.
    
    # Let's just do random search with prime checking
    
    found_prime_sumset = False
    
    # Random search for prime sumsets
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        if check_prime_sumset(A, B):
            best_A = A
            best_B = B
            best_score = 9
            found_prime_sumset = True
            break
    
    if found_prime_sumset:
        return [best_A, best_B]
    
    # Strategy 2: If no prime sumset found, do local search to maximize cardinality
    def local_search(A, B, deadline):
        nonlocal best_A, best_B, best_score
        current_A = list(A)
        current_B = list(B)
        current_score = score_pair(current_A, current_B)[1]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in current_A and new_val != current_A[i]:
                        continue
                    trial_A = list(current_A)
                    trial_A[i] = new_val
                    if len(set(trial_A)) < 3:
                        continue
                    valid, sc = score_pair(trial_A, current_B)
                    if valid and sc > current_score:
                        current_A = trial_A
                        current_score = sc
                        improved = True
                        break
                if improved:
                    break
            for i in range(3):
                if time.time() >= deadline:
                    break
                for new_val in range(1, 129):
                    if new_val in current_B and new_val != current_B[i]:
                        continue
                    trial_B = list(current_B)
                    trial_B[i] = new_val
                    if len(set(trial_B)) < 3:
                        continue
                    valid, sc = score_pair(current_A, trial_B)
                    if valid and sc > current_score:
                        current_B = trial_B
                        current_score = sc
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_A = list(current_A)
            best_B = list(current_B)
            best_score = current_score
        return current_score
    
    # Start local search from seed
    local_search([1, 7, 13], [4, 10, 16], deadline)
    
    # Try more random starts
    if time.time() < deadline:
        for attempt in range(200000):
            if time.time() >= deadline:
                break
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            valid, sc = score_pair(A, B)
            if valid and sc > best_score:
                best_A = list(A)
                best_B = list(B)
                best_score = sc
                local_search(best_A, best_B, deadline)
    
    # Strategy 3: Targeted search with A = {1, p, q} for primes p, q
    if time.time() < deadline:
        small_primes = [p for p in primes if p <= 128]
        for a in small_primes:
            if time.time() >= deadline:
                break
            for b in small_primes:
                if b <= a:
                    continue
                if time.time() >= deadline:
                    break
                A = [1, a, b]
                for _ in range(50):
                    B = sorted(random.sample(range(1, 129), 3))
                    valid, sc = score_pair(A, B)
                    if valid and sc > best_score:
                        best_A = list(A)
                        best_B = list(B)
                        best_score = sc
                        local_search(best_A, best_B, deadline)
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
