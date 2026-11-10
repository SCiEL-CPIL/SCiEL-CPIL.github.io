"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    # The problem asks for a set of distinct integers from 1..256 
    # such that every pair is coprime, maximizing the sum.
    
    # Insight:
    # If a number n is in the set, it consumes all its prime factors.
    # For any prime p, at most one number in the set can be divisible by p.
    # To maximize the sum, for each prime p, we should choose the largest 
    # multiple of p that is <= 256 and does not contain any other prime 
    # that is "reserved" for a larger value.
    
    # However, consider the trade-off:
    # If we pick p^k (largest power of p <= 256), we use up prime p.
    # If we pick p*q, we use up primes p and q.
    # Since p^k >= p and q^m >= q, and typically p^k + q^m >> p*q 
    # (especially for small primes where powers are large), it is strictly 
    # better to pick the maximum prime power for each prime individually 
    # rather than combining primes into composites.
    
    # Proof sketch:
    # Let S be the set of maximum prime powers for all primes p <= 256.
    # S = { p^k | p is prime, p^k <= 256 < p^(k+1) }
    # All elements in S are pairwise coprime because they have distinct prime bases.
    # Can we add any other number x?
    # Any x > 1 has a prime factor p. The element p^k in S shares factor p with x.
    # Thus gcd(x, p^k) > 1. So no other number can be added.
    # Can we replace p^k with a larger number x divisible by p?
    # x must be coprime to all q^m in S for q != p.
    # This implies x cannot have any prime factor q != p.
    # So x must be a power of p, i.e., x = p^j.
    # Since p^k is the largest power of p <= 256, x <= p^k.
    # So we cannot increase the sum by replacing.
    
    # Therefore, the optimal set is {1} union { p^k | p is prime <= 256 }.
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    result = [1] # 1 is coprime to everything and adds to the sum
    
    for p in primes:
        power = p
        while power * p <= MAX_N:
            power *= p
        result.append(power)
        
    # Ensure types are correct and distinct
    # The logic above guarantees distinctness because each prime p generates 
    # a unique power p^k, and different primes generate numbers with different 
    # prime factors, hence different values (except potentially 1, but p>=2).
    # Actually, could p^k == q^m? No, unique factorization.
    
    return result
# EVOLVE-BLOCK-END
