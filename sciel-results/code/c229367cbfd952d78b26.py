import time
import math

def construct():
    deadline = time.time() + 85
    
    TARGET = 123457
    
    # Generate primes up to TARGET using sieve
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(TARGET)
    
    # Initialize with 1 (which is 1-smooth)
    smooths = {1}
    
    # Iterate K through primes in increasing order
    for k in primes:
        if time.time() > deadline:
            break
            
        # Incrementally add k-smooth numbers
        # New numbers are of form m * k^j where m is in smooths (before adding k)
        # and j >= 1
        # We iterate over a copy of smooths because we are updating it
        # Actually, we can just generate new ones and add them.
        # To avoid using newly added k-multiples to generate more k-multiples (which is redundant 
        # as k^j * m where m is k-smooth is just another k-smooth number), 
        # we should only multiply numbers that did NOT have k as a factor in the previous step?
        # No, if we multiply a number that already has k, we get higher powers of k.
        # e.g. smooths has 2. p=2.
        # 2*2=4. 4*2=8.
        # If we multiply 4 (which was just added) by 2, we get 8.
        # If we iterate over the *original* smooths (without k factors), we get 2*2=4.
        # Then 4 is in the new set.
        # Do we need to generate 8 from 4?
        # 8 is 2^3.
        # If we start with 1.
        # p=2.
        # 1*2=2, 1*4=4, 1*8=8...
        # This generates all powers.
        # If we start with smooths={1, 2, 4} (from previous steps? No, p=2 is first).
        # If p=3.
        # smooths has {1, 2, 4, 8...}.
        # We generate 1*3=3, 1*9=9...
        # 2*3=6, 2*9=18...
        # 4*3=12...
        # This covers all combinations.
        
        # However, iterating over `smooths` while adding to it is dangerous in Python if we use `for s in smooths`.
        # We must iterate over a snapshot.
        # But taking a snapshot of a large set is expensive.
        # Alternative: collect new numbers in a separate set, then update.
        
        new_numbers = set()
        # We only need to consider base numbers from the *current* smooths set.
        # Any number generated is s * k^j.
        # If s is in smooths, s * k^j is k-smooth.
        # We don't need to generate numbers from numbers that were *just* added in this iteration,
        # because s * k^j where s = s' * k^i implies s * k^j = s' * k^(i+j),
        # which would be generated from s' with power i+j.
        # So we can iterate over the set *before* adding new numbers.
        
        # To avoid copying the whole set, we can just use `list(smooths)` if it's small.
        # Or, we can note that we only need to multiply by k, k^2, etc.
        # The set size is small enough (< 100k) that `list(smooths)` is fine.
        
        current_smooths = list(smooths)
        for s in current_smooths:
            v = s * k
            while v <= TARGET:
                new_numbers.add(v)
                v *= k
                
        if new_numbers:
            smooths.update(new_numbers)
            
        # Check for solution
        found = None
        # We can optimize the check by iterating over the smaller of smooths and 
        # the complement? No, just iterate smooths.
        # To break early, we just find one.
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                found = (a, b)
                break
                
        if found:
            return list(found)
            
    # Fallback
    return [1, TARGET - 1]
