import time
import math

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Generate primes up to target using simple sieve
    # We need primes up to target for factorization
    is_prime = bytearray(b'\x01') * (target + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(target)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, target + 1) if is_prime[i]]
    
    # Function to get largest prime factor of n
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
        if n > 1:
            return n
        else:
            # n was reduced to 1, so the largest prime factor is the last one we divided by
            # We need to track this properly
            pass
        # This approach is flawed for tracking the largest. Let's redo.
        return 1  # placeholder
    
    # Better approach: precompute largest prime factor using sieve
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                # Actually, we want the LARGEST prime factor, so we should overwrite
                # But the sieve above sets it to the first (smallest) prime factor
                # Let's fix: iterate and always set to current prime, so last one wins
                lpf[j] = i  # This will end up with the largest prime factor
    
    # Wait, the above sieve for lpf[j] = i in the inner loop will indeed set lpf[j] to the largest prime factor
    # because we iterate i from 2 to target, and for each prime i, we set lpf[j] = i for all multiples j.
    # The last prime that divides j will be the largest one.
    # But we need to be careful: we only want to do this for prime i.
    # Let's redo with proper prime checking.
    
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime (lpf[i] is still 0 means no smaller prime divided it)
            for j in range(i, target + 1, i):
                lpf[j] = i  # keep overwriting, so final value is largest prime factor
    
    # Now scan for best a, b with a + b = target, minimizing max(lpf[a], lpf[b])
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    half_target = target // 2
    for a in range(2, half_target + 1):
        if time.time() > deadline:
            break
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
        if best_score == 2:
            break
    
    return [int(best_a), int(best_b)]
