"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

TARGET = 123457

def linear_sieve(n):
    """Generate all primes up to n using linear sieve."""
    is_prime = [True] * (n + 1)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes

def construct():
    # Precompute all primes up to TARGET
    primes = linear_sieve(TARGET)
    
    # Set of K-smooth numbers (initially 1-smooth, i.e., just {1})
    smooth_set = {1}
    
    for p in primes:
        # Generate new smooth numbers by multiplying existing ones by powers of p
        # We must iterate over a snapshot of the set BEFORE adding new multiples of p
        # to ensure we don't miss combinations or do redundant work?
        # Actually, if we iterate over the current set and add s*p, s*p*p, etc.,
        # we need to be careful not to use newly added numbers to generate more numbers
        # in the same pass if we want to be strictly "multiplying old smooth numbers by p^k".
        # But smooth numbers are closed under multiplication by p.
        # If s is smooth (using primes <= p_prev), then s*p is smooth (using primes <= p).
        # If we add s*p to the set, and then later in the same loop we see s*p,
        # we would generate s*p*p. This is correct.
        # However, iterating over a growing set is dangerous in Python if we modify it.
        # We are using a set and updating it.
        
        # Efficient way:
        # Let S_old be the set of smooth numbers using primes < p.
        # New numbers are { s * p^k | s in S_old, k >= 1, s*p^k <= TARGET }
        
        # To get S_old, we can just take the current smooth_set before we start adding p-multiples.
        # But wait, smooth_set currently contains numbers using primes < p.
        # So we can iterate over the current smooth_set.
        
        # We need to add all s * p^k.
        # We can do this by iterating over the current smooth_set (which is S_old)
        # and for each s, multiply by p repeatedly.
        
        # Optimization: If s > TARGET // p, then s*p > TARGET, so no multiples.
        
        new_numbers = set()
        for s in smooth_set:
            v = s * p
            while v <= TARGET:
                new_numbers.add(v)
                v *= p
        
        if not new_numbers:
            # If no new numbers can be formed, adding this prime won't expand the set
            # (This happens if p > TARGET, but we sieve up to TARGET, so p <= TARGET.
            # If s=1, p <= TARGET, so 1*p is always added. So new_numbers is never empty for p <= TARGET)
            pass
            
        smooth_set.update(new_numbers)
        
        # Check for solution
        # We need a, b in smooth_set such that a + b = TARGET
        # Iterate through smooth_set
        for a in smooth_set:
            b = TARGET - a
            if b in smooth_set:
                # Ensure a, b are positive (they are, as smooth numbers are >= 1)
                return [a, b]
                
    # Fallback
    return [1, TARGET - 1]

# EVOLVE-BLOCK-END
