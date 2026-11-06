"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    for p in primes:
        if p > start + length - 1 and p * p > start + length - 1:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > start + length - 1:
            continue
        
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest contiguous subarray with all distinct values."""
    n = len(phis)
    if n == 0:
        return 0
    best = 1
    left = 0
    seen = {}
    for right, val in enumerate(phis):
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        current_len = right - left + 1
        if current_len > best:
            best = current_len
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    random.seed(42)
    
    # Strategy 1: Random search for full length-200 runs
    # Try many random starts
    for _ in range(300):
        if time.time() > deadline - 40:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        run_len = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            # Find the actual start of this best run
            left = 0
            seen = {}
            for right, val in enumerate(phis):
                if val in seen and seen[val] >= left:
                    left = seen[val] + 1
                seen[val] = right
                if right - left + 1 == run_len:
                    best_start = s + left
                    break
            else:
                # Fallback: find first position where run_len consecutive are distinct
                for i in range(len(phis) - run_len + 1):
                    if len(set(phis[i:i+run_len])) == run_len:
                        best_start = s + i
                        break
    
    # Strategy 2: Try structured starts near primorials
    if time.time() < deadline - 20:
        primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230, 200862915270]
        for pm in primorials:
            if time.time() > deadline - 15:
                break
            # Try a range of multiples
            num_mults = min(50, max(1, int((max_n - max_k) / pm)))
            for mult in random.sample(range(1, num_mults + 1), min(20, num_mults)):
                if time.time() > deadline - 15:
                    break
                base = pm * mult
                if base < max_k or base + max_k > max_n:
                    continue
                for offset in random.sample(range(-50, 51), 11):
                    s = base + offset
                    if s < 1 or s + max_k - 1 > max_n:
                        continue
                    phis = compute_phi_segment(s, max_k, primes)
                    if len(set(phis)) == max_k:
                        return {'start': s, 'length': max_k}
                    run_len = find_longest_distinct_run(phis)
                    if run_len > best_k:
                        best_k = run_len
                        left = 0
                        seen = {}
                        for right, val in enumerate(phis):
                            if val in seen and seen[val] >= left:
                                left = seen[val] + 1
                            seen[val] = right
                            if right - left + 1 == run_len:
                                best_start = s + left
                                break
    
    # Strategy 3: Targeted search in different magnitude ranges
    if time.time() < deadline - 10:
        ranges = [
            (10**4, 10**6),
            (10**6, 10**8),
            (10**8, 10**10),
            (10**10, int(max_n - max_k)),
        ]
        for lo, hi in ranges:
            if time.time() > deadline - 10:
                break
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                s = random.randint(lo, min(hi, max_n - max_k + 1))
                phis = compute_phi_segment(s, max_k, primes)
                if len(set(phis)) == max_k:
                    return {'start': s, 'length': max_k}
                run_len = find_longest_distinct_run(phis)
                if run_len > best_k:
                    best_k = run_len
                    left = 0
                    seen = {}
                    for right, val in enumerate(phis):
                        if val in seen and seen[val] >= left:
                            left = seen[val] + 1
                        seen[val] = right
                        if right - left + 1 == run_len:
                            best_start = s + left
                            break
    
    # Strategy 4: If we found a good partial run, try to extend it
    if best_k >= 10 and time.time() < deadline - 5:
        # Try extending the best found run
        for delta in range(-100, 101):
            if time.time() > deadline - 3:
                break
            s = best_start + delta
            if s < 1 or s + max_k - 1 > max_n:
                continue
            phis = compute_phi_segment(s, max_k, primes)
            if len(set(phis)) == max_k:
                return {'start': s, 'length': max_k}
            run_len = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                left = 0
                seen = {}
                for right, val in enumerate(phis):
                    if val in seen and seen[val] >= left:
                        left = seen[val] + 1
                    seen[val] = right
                    if right - left + 1 == run_len:
                        best_start = s + left
                        break
    
    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
