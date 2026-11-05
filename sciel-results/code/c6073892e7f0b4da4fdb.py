import time
import math

def construct():
    # Use a safer deadline to avoid timeout (evaluator kills at 90s)
    deadline = time.time() + 60
    
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
        new_numbers = set()
        current_smooths = list(smooths)
        for s in current_smooths:
            v = s * k
            while v <= TARGET:
                new_numbers.add(v)
                v *= k
                
        if new_numbers:
            # Check for solution involving new numbers
            # If a + b = TARGET, and a is new, b must be in smooths (old) or new_numbers (new)
            # Note: If b is in new_numbers, it might be the same as a (if a=b=TARGET/2) or different.
            # We just need to check membership.
            
            # Optimization: Iterate over the smaller set if possible? 
            # Usually new_numbers is smaller than smooths initially, but grows.
            # Checking membership in a set is O(1).
            
            # We can break early if found
            found = None
            for a in new_numbers:
                b = TARGET - a
                if b >= 1 and (b in smooths or b in new_numbers):
                    found = (a, b)
                    break
            
            if found:
                return list(found)
            
            smooths.update(new_numbers)
            
    # Fallback
    return [1, TARGET - 1]
