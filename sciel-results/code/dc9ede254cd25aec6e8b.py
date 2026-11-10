def construct():
    import time
    import math
    from collections import defaultdict
    
    start = time.time()
    deadline = start + 140.0
    
    # Fallback known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Helper: exact sigma for individual numbers via factorization
    def factorize(n):
        """Factorize n using trial division up to sqrt(n)."""
        if n == 1:
            return []
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors
    
    def sigma_exact(n):
        """Compute sigma(n) exactly."""
        if n == 1:
            return 1
        factors = factorize(n)
        result = 1
        for p, exp in factors:
            result *= (p ** (exp + 1) - 1) // (p - 1)
        return result
    
    # Phase 1: Count-only sieve pass to find M values with many preimages
    N_sieve = 8_000_000
    if time.time() < deadline - 60:
        # First pass: just count occurrences of each M
        m_counts = defaultdict(int)
        sigma_small = [0] * (N_sieve + 1)
        for i in range(1, N_sieve + 1):
            for j in range(i, N_sieve + 1, i):
                sigma_small[j] += i
        
        for k in range(2, N_sieve + 1):
            M = k * sigma_small[k]
            if M <= 10**18:
                m_counts[M] += 1
        
        # Find top candidates by count
        top_candidates = sorted(m_counts.items(), key=lambda x: -x[1])[:50]
        
        # Second pass: collect k values only for top candidates
        top_set = set(M for M, c in top_candidates if c > best_count)
        m_to_k = defaultdict(list)
        for k in range(2, N_sieve + 1):
            M = k * sigma_small[k]
            if M in top_set:
                m_to_k[M].append(k)
        
        for M, ks in m_to_k.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]
        
        del sigma_small
        del m_counts
        del m_to_k
    
    # Phase 2: For the best M found, enumerate all divisors to find more k
    if time.time() < deadline - 30 and best_M is not None:
        try:
            factors_M = factorize(best_M)
            divisors = [1]
            for p, exp in factors_M:
                new_divs = []
                for d in divisors:
                    for e in range(exp + 1):
                        new_divs.append(d * (p ** e))
                divisors = new_divs
            
            all_ks = []
            for k in divisors:
                if k <= math.isqrt(best_M) and k > 1:
                    try:
                        if k * sigma_exact(k) == best_M:
                            all_ks.append(k)
                    except Exception:
                        continue
            
            if len(all_ks) > best_count:
                best_count = len(all_ks)
                best_k = all_ks
        except Exception:
            pass
    
    # Phase 3: Smooth number search for larger k values
    if time.time() < deadline - 20:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        max_smooth = 20_000_000_000
        
        def generate_smooth_numbers(primes, max_val):
            results = set()
            def gen(idx, current):
                if current > max_val:
                    return
                if idx == len(primes):
                    if current > 1:
                        results.add(current)
                    return
                p = primes[idx]
                gen(idx + 1, current)
                val = current * p
                while val <= max_val:
                    gen(idx + 1, val)
                    val *= p
            gen(0, 1)
            return list(results)
        
        smooth_nums = generate_smooth_numbers(primes, max_smooth)
        smooth_nums.sort()
        
        m_to_k2 = defaultdict(list)
        for k in smooth_nums:
            if time.time() > deadline - 10:
                break
            if k <= N_sieve:
                continue
            try:
                sig = sigma_exact(k)
                M = k * sig
                if M <= 10**18 and k <= math.isqrt(M):
                    m_to_k2[M].append(k)
            except Exception:
                continue
        
        for M, ks in m_to_k2.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]
        
        # Also check if smooth numbers add to existing best_M
        if best_M in m_to_k2:
            combined = list(set(best_k + m_to_k2[best_M]))
            if len(combined) > best_count:
                best_count = len(combined)
                best_k = combined
    
    # Final verification
    if best_M is not None and best_k:
        valid_k = []
        for k in best_k:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if k < 1 or k > math.isqrt(best_M):
                continue
            try:
                if k * sigma_exact(k) == best_M:
                    valid_k.append(k)
            except Exception:
                continue
        
        if len(valid_k) >= 2:
            best_k = valid_k
            best_count = len(valid_k)
        else:
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3
    
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}
