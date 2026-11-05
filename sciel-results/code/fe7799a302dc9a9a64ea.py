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
    
    def get_smooths(k, limit):
        """Generate all numbers <= limit whose prime factors are all <= k."""
        ps = [p for p in primes if p <= k]
        smooths = {1}
        for p in ps:
            new = set()
            for s in smooths:
                v = s
                while v * p <= limit:
                    v *= p
                    new.add(v)
            smooths |= new
        return smooths

    # Iterate K through primes in increasing order
    # This is more efficient than binary search if the optimal K is small
    # We stop at the first K that yields a solution
    # We can limit the search to reasonable K values
    # If no solution found for K up to 100, we can fall back to binary search or just return best effort
    
    for k in primes:
        if time.time() > deadline:
            break
            
        smooths = get_smooths(k, TARGET)
        
        found = None
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                found = (a, b)
                break
        
        if found:
            return list(found)
            
        # If k is large enough, we might have missed something or it's just not there
        # But since we are going in increasing order, the first found is the optimal max prime factor
        # We should probably stop if k exceeds some bound, but 60s is plenty for small k
        
    # Fallback if no small prime solution found
    # Try binary search as before, or just return a known good pair
    # Since the previous program scored 58.45 (K=11), it's likely K=11 is the answer.
    # If the loop above didn't find it, something is wrong.
    # Let's force check K=11, 13, 17 etc if the loop failed? 
    # Actually the loop covers all primes.
    
    # Last resort
    return [1, 123456]
