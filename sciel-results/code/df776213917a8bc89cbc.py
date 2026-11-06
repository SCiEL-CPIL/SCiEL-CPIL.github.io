import time
import math
import sys

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def sieve(n):
    """Return a list of primes up to n."""
    if n < 2:
        return []
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n+1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We expect the best k to be found for relatively small p.
    # Let's sieve primes up to 2,000,000.
    # pi(2e6) is approx 148,933.
    # This might take a bit of memory and time, but is feasible.
    # Let's start with 1,000,000 to be safe on time, and extend if fast.
    
    limit = 1000000
    primes = sieve(limit)
    
    best_k = 0
    best_p = 3
    
    # Helper to check if n is p-smooth
    # primes is the global list of primes up to limit
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        rem = n
        # We need primes <= p
        # Since p is in our sieve list, we can find its index
        # But iterating all primes <= p is O(pi(p))
        # Optimization: if p is large, we can't iterate all.
        # But for p <= 1e6, pi(p) <= 78k.
        # We break early if q*q > rem.
        
        # Find index of p in primes? Or just iterate and break when q > p.
        # Since primes is sorted, we can use binary search to get the slice, 
        # but iterating is fine if we break early.
        
        for q in primes:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
        
        if rem == 1:
            return True
        
        # If we broke because q > p, then rem has a factor > p
        # Check if the last q was > p or q*q > rem
        # If q > p, then rem is not 1 and has factors > p.
        # If q*q > rem, then rem is prime.
        
        # We need to know why we broke.
        # Let's re-evaluate.
        
        # Actually, the loop above breaks if q > p OR q*q > rem.
        # If it breaks because q > p, it means we exhausted all primes <= p.
        # If rem > 1, it has a prime factor > p.
        # If it breaks because q*q > rem, it means rem is prime.
        # We need to check if rem <= p.
        
        # How to distinguish?
        # We can check if the last q that was processed was > p? No, we break before processing.
        
        # Let's use a flag.
        # Or simpler:
        # After the loop, if rem > 1:
        # If the loop terminated because q > p, then rem is not p-smooth.
        # If the loop terminated because q*q > rem, then rem is prime.
        # We can check if is_prime(rem) and rem <= p.
        # But if rem is composite and we broke because q > p, is_prime(rem) is false (or we don't care).
        # If rem is prime and we broke because q > p, it means rem > p (since q <= p were checked and didn't divide, and q reached p+something? No, q is the next prime).
        # If rem is prime, and we broke because q > p, then rem must be >= q > p. So rem > p.
        # So in both cases, if rem > 1 and we broke because q > p, it's not smooth.
        
        # What if we broke because q*q > rem?
        # Then rem is prime.
        # We need to check if rem <= p.
        
        # So:
        # If rem > 1:
        #    If is_prime(rem):
        #       return rem <= p
        #    Else:
        #       # rem is composite.
        #       # This can only happen if we broke because q > p?
        #       # If we broke because q*q > rem, rem is prime.
        #       # So if rem is composite, we must have broken because q > p.
        #       # In that case, rem has a factor > p.
        #       return False
        #       # Wait, what if rem is a square of a prime > p?
        #       # Then q*q > rem would not have triggered if q <= p.
        #       # If q > p, we break. rem is composite. Correct.
        
        # However, calling is_prime(rem) is expensive if rem is large.
        # But rem < n < p^2 + 200.
        # If p is small, rem is small.
        
        # Let's refine the loop to track the break reason.
        # Actually, simpler logic:
        # Trial divide by primes <= min(p, sqrt(n)).
        # If rem > 1:
        #    If rem <= p and is_prime(rem): return True
        #    Else: return False
        
        # Wait, if rem is composite and <= p, it is p-smooth.
        # Example: p=10, n=49. rem=49. 49<=10 is false.
        # Example: p=10, n=25. rem=25. 25<=10 is false.
        # Example: p=10, n=4. rem=4. 4<=10 true. is_prime(4) false.
        # But 4 is 10-smooth.
        
        # So the logic "rem <= p and is_prime(rem)" is insufficient.
        # We need to fully factor rem or check if all factors <= p.
        
        # Given the constraints, let's just do:
        # If rem > 1:
        #    # Check if rem is p-smooth
        #    # Since we divided by all primes <= p (up to sqrt(rem) or p),
        #    # if we broke because q > p, rem has factor > p.
        #    # if we broke because q*q > rem, rem is prime.
        
        # Let's just use the fact that if we broke because q > p, it's not smooth.
        # If we broke because q*q > rem, rem is prime.
        
        # How to know?
        # We can check if the last q in the loop was > p.
        
        # Let's rewrite is_p_smooth for clarity and correctness.
        
        pass # Placeholder

    # Redefine is_p_smooth correctly
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        rem = n
        broke_due_to_p = False
        for q in primes:
            if q > p:
                broke_due_to_p = True
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
        
        if rem == 1:
            return True
        
        if broke_due_to_p:
            # We exhausted all primes <= p.
            # rem > 1 implies rem has a prime factor > p.
            return False
        
        # We broke because q*q > rem.
        # rem is prime.
        return rem <= p

    # Scan primes
    for p in primes:
        if time.time() > deadline:
            break
        
        base = p * p
        k = 0
        # Check up to 200, or until deadline
        for i in range(1, 201):
            if time.time() > deadline - 5:
                break
            n = base + i
            if is_p_smooth(n, p):
                k = i
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we have time, try random larger primes
    # But the systematic scan up to 1e6 should be sufficient to beat k=3.
    
    return {'p': best_p, 'k': best_k}
