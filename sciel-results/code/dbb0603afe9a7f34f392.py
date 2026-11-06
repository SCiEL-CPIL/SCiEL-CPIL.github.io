import time

def construct():
    N = 123457
    deadline = time.time() + 85.0
    
    # Generate all 11-smooth numbers up to N
    # 11-smooth: all prime factors in {2, 3, 5, 7, 11}
    smooth = set()
    
    def gen_smooth(powers, primes, current, limit):
        if time.time() > deadline:
            return
        if current > limit:
            return
        if primes:
            p = primes[0]
            rest = primes[1:]
            val = current
            while val <= limit:
                smooth.add(val)
                if rest:
                    gen_smooth(powers, rest, val, limit)
                else:
                    # Last prime, just add
                    pass
                val *= p
        else:
            smooth.add(current)
    
    # Better approach: iterative generation
    smooth = set()
    primes = [2, 3, 5, 7, 11]
    
    def gen(idx, current):
        if time.time() > deadline:
            return
        if current > N:
            return
        if idx == len(primes):
            smooth.add(current)
            return
        p = primes[idx]
        val = current
        while val <= N:
            gen(idx + 1, val)
            val *= p
    
    gen(0, 1)
    
    smooth_list = sorted(smooth)
    smooth_set = set(smooth_list)
    
    best_a = None
    best_b = None
    best_max_pf = N  # upper bound
    
    # For each smooth number a, check if N-a is also smooth
    for a in smooth_list:
        b = N - a
        if b <= 0:
            break
        if b in smooth_set:
            # Both are 11-smooth, so max(P(a),P(b)) <= 11
            # Find actual largest prime factor
            def largest_prime_factor(n):
                if n == 1:
                    return 1
                pf = 1
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        pf = d
                        n //= d
                    d += 1
                if n > 1:
                    pf = n
                return pf
            
            pa = largest_prime_factor(a)
            pb = largest_prime_factor(b)
            mx = max(pa, pb)
            if mx < best_max_pf:
                best_max_pf = mx
                best_a = a
                best_b = b
    
    if best_a is not None:
        return [best_a, best_b]
    
    # Fallback: brute force search for minimal max(P(a), P(b))
    # Try increasing threshold T and check if there exists a such that
    # both a and N-a have all prime factors <= T
    
    def largest_prime_factor_fast(n):
        if n == 1:
            return 1
        pf = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf = d
                n //= d
            d += 1
        if n > 1:
            pf = n
        return pf
    
    # Brute force with early termination
    best_max_pf = N
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        pa = largest_prime_factor_fast(a)
        if pa >= best_max_pf:
            continue
        pb = largest_prime_factor_fast(b)
        mx = max(pa, pb)
        if mx < best_max_pf:
            best_max_pf = mx
            best_a = a
            best_b = b
            if best_max_pf <= 11:
                break
    
    return [best_a, best_b]
