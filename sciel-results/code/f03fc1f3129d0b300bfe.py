import math

def construct():
    N = 123457
    # Sieve for smallest prime factor
    spf = list(range(N))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i*i, N, i):
                if spf[j] == j:
                    spf[j] = i

    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            if p > max_p:
                max_p = p
            while n % p == 0:
                n //= p
        return max_p

    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        cur_val = pa if pa > pb else pb
        if cur_val < best_val:
            best_val = cur_val
            best_a = a
            best_b = b
            # If we find a very small value, we can stop early?
            # The minimum possible is 2 (if both are powers of 2, sum must be power of 2, 123457 is odd, so one is 1, other is 123456 which is not power of 2).
            # Actually, if max_val becomes 2, we can stop. But it's unlikely.
            
    return [best_a, best_b]
