import time
import math
from itertools import product

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Fallback seed result
    fallback = {'M': 333312, 'k': [336, 372, 434]}

    # Strategy: Generate k from smooth numbers (small primes with bounded exponents)
    # and find M collisions where multiple k give same k*sigma(k)
    
    # Small primes to use
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # For each prime, determine max exponent such that p^e stays reasonable
    # We want k values up to ~10^9 or so to stay within M <= 10^18
    # sigma(k) is roughly O(k), so k*sigma(k) ~ k^2, so k up to ~10^9 gives M up to ~10^18
    
    # Determine exponent bounds per prime
    exp_bounds = {}
    for p in primes:
        e = 0
        val = 1
        while val * p <= 10**9:
            val *= p
            e += 1
        exp_bounds[p] = e
    
    # Generate all smooth k values
    # Use recursive generation to avoid huge product spaces
    smooth_ks = set()
    
    def gen_smooth(idx, current_val):
        if time_left() < 30:
            return
        if idx == len(primes):
            if current_val > 1:
                smooth_ks.add(current_val)
            return
        p = primes[idx]
        max_e = exp_bounds[p]
        for e in range(max_e + 1):
            new_val = current_val * (p ** e)
            if new_val > 10**9:
                break
            gen_smooth(idx + 1, new_val)
    
    gen_smooth(0, 1)
    
    # Convert to sorted list for processing
    k_list = sorted(smooth_ks)
    
    # Function to compute sigma(n) via trial division for individual numbers
    def sigma(n):
        if n <= 1:
            return n
        result = 1
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                # d is a prime factor
                power_sum = 1
                power = 1
                while temp % d == 0:
                    temp //= d
                    power *= d
                    power_sum += power
                result *= power_sum
            d += 1 if d == 2 else 2
        if temp > 1:
            result *= (1 + temp)
        return result
    
    # Compute M = k * sigma(k) for each smooth k and group by M
    m_to_ks = {}
    for k in k_list:
        if time_left() < 25:
            break
        s = sigma(k)
        M = k * s
        if M > 10**18:
            continue
        if M in m_to_ks:
            m_to_ks[M].append(k)
        else:
            m_to_ks[M] = [k]
    
    # Find the M with most solutions
    best_M = 0
    best_count = 0
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
    
    # If we found something better than seed (3 solutions), use it
    if best_count > 3 and best_M > 0:
        # Verify each k satisfies constraints
        valid_ks = []
        isqrt_M = math.isqrt(best_M)
        for k in m_to_ks[best_M]:
            if k <= isqrt_M and k * sigma(k) == best_M:
                valid_ks.append(k)
        if len(valid_ks) > 3:
            return {'M': best_M, 'k': valid_ks}
    
    # Fallback: also try a quick sieve for small k to find collisions
    N_sieve = 2_000_000
    if time_left() > 40:
        sig = [0] * (N_sieve + 1)
        for d in range(1, N_sieve + 1):
            if time_left() < 30:
                break
            for m in range(d, N_sieve + 1, d):
                sig[m] += d
        
        m_to_ks2 = {}
        for k in range(1, N_sieve + 1):
            M = k * sig[k]
            if M > 10**18:
                continue
            if M in m_to_ks2:
                m_to_ks2[M].append(k)
            else:
                m_to_ks2[M] = [k]
        
        # Merge with smooth results
        for M, ks in m_to_ks2.items():
            if M in m_to_ks:
                # Merge, keeping distinct
                existing = set(m_to_ks[M])
                for k in ks:
                    if k not in existing:
                        m_to_ks[M].append(k)
                        existing.add(k)
            else:
                m_to_ks[M] = ks
        
        # Re-find best
        best_M2 = 0
        best_count2 = 0
        for M, ks in m_to_ks.items():
            if len(ks) > best_count2:
                best_count2 = len(ks)
                best_M2 = M
        
        if best_count2 > 3 and best_M2 > 0:
            valid_ks = []
            isqrt_M = math.isqrt(best_M2)
            for k in m_to_ks[best_M2]:
                if k <= isqrt_M:
                    # Verify sigma
                    s = sigma(k)
                    if k * s == best_M2:
                        valid_ks.append(k)
            if len(valid_ks) > 3:
                return {'M': best_M2, 'k': valid_ks}
    
    return fallback
