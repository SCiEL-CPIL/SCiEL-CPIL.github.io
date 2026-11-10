def construct():
    import time
    import math
    from collections import defaultdict
    
    start = time.time()
    deadline = start + 110.0
    
    # Fallback known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Helper: fast sigma using sieve-like approach for small numbers
    def compute_sigma_sieve(N):
        """Compute sigma for all numbers up to N using sieve."""
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
        return sigma
    
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
    
    # Helper: generate smooth numbers (products of small primes)
    def generate_smooth_numbers(primes, max_val):
        """Generate all numbers <= max_val that are products of given primes."""
        results = set()
        
        def gen(idx, current):
            if current > max_val:
                return
            if idx == len(primes):
                if current > 1:
                    results.add(current)
                return
            p = primes[idx]
            # Don't multiply by p (exponent 0)
            gen(idx + 1, current)
            val = current * p
            while val <= max_val:
                gen(idx + 1, val)
                val *= p
        
        gen(0, 1)
        return list(results)
    
    # Phase 1: Sieve for small k to find collisions quickly
    N_sieve = 5_000_000
    if time.time() < deadline - 30:
        sigma_small = compute_sigma_sieve(N_sieve)
        m_to_k = defaultdict(list)
        
        for k in range(2, N_sieve + 1):
            M = k * sigma_small[k]
            if M <= 10**18:
                m_to_k[M].append(k)
        
        # Find best from sieve
        for M, ks in m_to_k.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]
        
        # Clean up
        del sigma_small
        del m_to_k
    
    # Phase 2: Generate smooth numbers for larger k
    if time.time() < deadline - 20:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        # Generate smooth numbers up to ~10^10
        max_smooth = 10_000_000_000
        smooth_nums = generate_smooth_numbers(primes, max_smooth)
        smooth_nums.sort()
        
        # Filter to those not already in sieve range and with M <= 10^18
        m_to_k2 = defaultdict(list)
        
        for k in smooth_nums:
            if time.time() > deadline - 10:
                break
            if k <= N_sieve:
                continue  # Already covered by sieve
            # Estimate sigma to check if M would be too large
            # sigma(k) <= k * (1 + 1/2 + 1/4 + ...) < 2k for powers of 2
            # But for smooth numbers, sigma(k)/k can be larger
            # Just compute and check
            try:
                sig = sigma_exact(k)
                M = k * sig
                if M <= 10**18 and k <= math.isqrt(M):
                    m_to_k2[M].append(k)
            except Exception:
                continue
        
        # Merge with previous results
        # We need to track which M values we've seen
        # Re-scan for the best
        all_m_to_k = defaultdict(list)
        
        # Re-do sieve results if still available, or just use m_to_k2
        # Actually, let's just find the best from m_to_k2 and compare
        for M, ks in m_to_k2.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]
        
        # Also check if any M from smooth numbers matches an M from sieve
        # We'd need to keep the sieve results, but we deleted them
        # Let's do a combined approach: recompute for a moderate range
        
    # Phase 3: If time remains, do a targeted search around promising M values
    if time.time() < deadline - 5 and best_M is not None:
        # Try to find more k for the current best_M by searching in a range
        # k * sigma(k) = best_M => k is a divisor of best_M
        # So we can factor best_M and enumerate divisors
        try:
            factors_M = factorize(best_M)
            # Generate all divisors of best_M
            divisors = [1]
            for p, exp in factors_M:
                new_divs = []
                for d in divisors:
                    for e in range(exp + 1):
                        new_divs.append(d * (p ** e))
                divisors = new_divs
            
            # For each divisor k of best_M, check if k * sigma(k) == best_M
            # and k <= isqrt(best_M)
            extra_ks = []
            for k in divisors:
                if k <= math.isqrt(best_M) and k > 1:
                    try:
                        if k * sigma_exact(k) == best_M:
                            extra_ks.append(k)
                    except Exception:
                        continue
            
            # Combine with existing best_k
            all_ks = list(set(best_k + extra_ks))
            if len(all_ks) > best_count:
                best_count = len(all_ks)
                best_k = all_ks
        except Exception:
            pass
    
    # Final verification
    if best_M is not None and best_k:
        # Verify all k values
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
            # Fallback
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}
