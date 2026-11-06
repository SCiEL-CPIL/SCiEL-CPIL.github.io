import time
import math

def construct():
    start_time = time.time()
    deadline = 115

    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Generate primes up to 10^7
    primes = sieve(10**7)
    
    best_k = 0
    best_p = 2
    
    # Check p=2, 3, 5...
    # We can skip very small p if we know they fail?
    # p=7 works with k=1.
    
    for p in primes:
        if time.time() - start_time > deadline - 5:
            break
            
        k = 0
        # Check up to 200
        for i in range(1, 201):
            n = p * p + i
            # Check if n is p-smooth
            # Divide by primes q <= p
            # Optimization: if q*q > n, break
            # If n > 1 at end, it has a factor > p (or is prime > p)
            
            # We only need to divide by q <= p
            # But if q > sqrt(n), we can stop.
            # If n is not 1 and q > sqrt(n), then n is prime.
            # If n is prime, is it <= p?
            # n ~ p^2, so n > p. So it fails.
            
            limit_q = min(p, int(math.isqrt(n)))
            
            # Find index of last prime <= limit_q
            # Since primes is sorted, we can iterate and break
            smooth = True
            for q in primes:
                if q > limit_q:
                    break
                if n % q == 0:
                    while n % q == 0:
                        n //= q
                    if n == 1:
                        break
                # If n becomes 1, we can stop
                if n == 1:
                    break
            
            if n == 1:
                k += 1
            else:
                # n > 1
                # If n <= p, it is smooth?
                # No, we only divided by q <= limit_q.
                # If limit_q < p, we didn't check primes in (limit_q, p].
                # But if n <= p, and n is not 1, then n has a prime factor <= n <= p.
                # Wait.
                # If we stopped because q > sqrt(n), then n is prime.
                # If n is prime and n <= p, it is smooth.
                # If n is prime and n > p, it is not smooth.
                # If we stopped because q > p, then we checked all q <= p.
                # If n > 1, it has a factor > p. Not smooth.
                
                if n <= p:
                    # n is prime (since we checked up to sqrt(n) or p)
                    # Actually, if limit_q == p, we checked all q <= p.
                    # If n > 1, it has a factor > p?
                    # No, if we checked all q <= p, and n > 1, then n has no factor <= p.
                    # So n has a factor > p.
                    # So n is NOT smooth.
                    # Wait.
                    # If n is prime and n <= p, then n is a factor <= p.
                    # Did we check n?
                    # If n <= p, then n <= limit_q (if limit_q=p) or n <= sqrt(n) (if limit_q=sqrt(n)).
                    # If n is prime, it is not divisible by any q < n.
                    # So the loop `while n % q == 0` never triggers for q < n.
                    # So n remains n.
                    # If n <= p, it is smooth.
                    # If n > p, it is not smooth.
                    
                    if n <= p:
                        k += 1
                    else:
                        smooth = False
                else:
                    smooth = False
                    
            if not smooth:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
