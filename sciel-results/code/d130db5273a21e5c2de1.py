import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Precompute small primes for trial division and segmented sieve ---
    LIMIT_TRIAL = 10**5
    is_prime_small = bytearray(b'\x01') * (LIMIT_TRIAL + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(math.isqrt(LIMIT_TRIAL)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = bytearray(len(is_prime_small[i*i::i]))
    small_primes = [i for i in range(2, LIMIT_TRIAL + 1) if is_prime_small[i]]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < LIMIT_TRIAL:
            return bool(is_prime_small[n])
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi_large(n):
        if n == 1:
            return 1
        temp = n
        result = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                factors = factorize(temp)
                for p in set(factors):
                    result -= result // p
        return result

    def phi_batch_segmented(start, count):
        """
        Compute phi for count consecutive integers starting at start using segmented sieve.
        This is much faster than individual factorization for large numbers.
        """
        if count <= 0:
            return []
        
        # Initialize arrays for the segment
        # We need to store the remaining value to factorize later if > 1
        # and the current phi value (initialized to n)
        # Actually, for segmented sieve of phi:
        # phi[i] = i initially.
        # For each prime p, for multiples j of p in the segment:
        #   phi[j] = phi[j] // p * (p-1)
        
        # However, we only have small primes up to LIMIT_TRIAL.
        # Numbers > LIMIT_TRIAL^2 might have large prime factors.
        # So we do:
        # 1. Sieve with small primes.
        # 2. For remaining values > 1, factorize them (they are products of large primes).
        
        nums = np.arange(start, start + count, dtype=np.int64)
        # We need to be careful with int64 overflow if start is close to 10^12.
        # phi(n) <= n, so int64 is safe.
        
        # Copy nums to phi_arr
        phi_arr = nums.copy()
        
        # Segmented sieve with small primes
        for p in small_primes:
            if p > math.isqrt(start + count):
                # Optimization: if p*p > max(nums), then p can only divide nums if num==p.
                # But we are sieving multiples.
                # Actually, standard segmented sieve iterates primes up to sqrt(max_val).
                # If p > sqrt(max_val), then p*p > max_val, so p can appear at most once in factorization.
                # But wait, if p > sqrt(max_val), then p*p > max_val.
                # So any multiple of p in the range is p * k where k < p.
                # Since we are sieving multiples, we just need to find multiples.
                # The loop condition for p should be p*p <= max_val.
                # If p*p > max_val, we still need to check if p itself is in the range?
                # No, the sieve updates multiples.
                # If p*p > max_val, then the only multiple of p in [start, start+count] could be p itself (if start <= p <= start+count).
                # But if p > sqrt(max_val), then p is prime (from small_primes list).
                # If p is in the range, phi[p] should be p-1.
                # Our loop:
                # start_idx = max(p, ceil(start/p)*p)
                # If p*p > max_val, then start_idx will be p if p >= start.
                # So we should not break early based on p*p > max_val if we want to catch primes in the range.
                # However, small_primes only goes up to 10^5.
                # sqrt(10^12) = 10^6.
                # So we need primes up to 10^6 for a complete segmented sieve without residual factorization.
                # Since we only have primes up to 10^5, we MUST do residual factorization for remaining factors.
                
                # So we should iterate ALL small primes, or at least those that could have multiples in the range.
                # A prime p has a multiple in [start, start+count] if there exists k such that k*p in [start, start+count].
                # This is possible for any p <= start+count.
                # But if p > start+count, no multiples.
                # So we break if p > start+count.
                
                pass # Don't break here, continue to next p if p is small enough to have multiples.
            
            if p > start + count:
                break
                
            # Find first multiple of p in [start, start+count]
            first = start
            if first % p != 0:
                first = first + (p - first % p)
            
            # Update phi_arr for indices corresponding to first, first+p, ...
            # Index in phi_arr is (num - start)
            # num = first, first+p, ...
            # idx = 0, p, 2p ... relative to first-start?
            # Let's use slicing.
            # Indices: (first - start), (first - start) + p, ...
            # Step is p.
            
            idx_start = first - start
            # We need to update phi_arr[idx_start::p]
            # phi[n] = phi[n] / p * (p-1)
            # phi_arr[idx_start::p] = phi_arr[idx_start::p] // p * (p-1)
            
            # Vectorized update
            phi_arr[idx_start::p] = (phi_arr[idx_start::p] // p) * (p - 1)
        
        # Now, for each number, if the remaining value (after removing small prime factors) is > 1,
        # it must be a prime or a product of large primes.
        # Wait, the sieve above updates phi_arr directly.
        # Does it leave a "remaining" factor?
        # No, phi_arr holds the current value of phi.
        # But we need to know if there are large prime factors left to handle.
        # The standard method:
        # Let rem[i] = start + i.
        # For each small prime p:
        #   for each multiple m of p:
        #     rem[m] //= p
        #     phi[m] = phi[m] // p * (p-1)
        # After processing all small primes, if rem[m] > 1, then rem[m] is a prime (or product of primes > LIMIT_TRIAL).
        # Since we used all primes up to LIMIT_TRIAL, any remaining factor is > LIMIT_TRIAL.
        # If rem[m] is prime, phi[m] = phi[m] // rem[m] * (rem[m]-1).
        # If rem[m] is composite, it's a product of large primes. We need to factorize it.
        
        # Let's re-implement phi_batch_segmented to track remainders.
        
        rem = np.arange(start, start + count, dtype=np.int64)
        phi_vals = rem.copy()
        
        for p in small_primes:
            if p > start + count:
                break
            first = start
            if first % p != 0:
                first = first + (p - first % p)
            idx_start = first - start
            
            # Update rem and phi_vals
            # rem[idx_start::p] //= p
            # phi_vals[idx_start::p] = phi_vals[idx_start::p] // p * (p-1)
            
            # We must be careful: rem changes, but we only need to know if it becomes 1 or stays > 1.
            # Actually, we just need to divide rem by p as many times as it appears.
            # But phi update is done once per distinct prime factor.
            # So we need to know if p divides rem.
            
            # Efficient way:
            # Check which elements in rem[idx_start::p] are divisible by p.
            # Since they are multiples of p by construction, they are all divisible by p.
            # But after dividing by p, are they still divisible by p?
            # We need to divide rem by p fully to check for large factors later?
            # No, for the phi formula, we only subtract once per distinct prime.
            # So we just need to divide phi by p*(p-1) once.
            # And we should divide rem by p once to see if it's still divisible?
            # Actually, if we divide rem by p, and it's still divisible by p, it's still a small prime factor.
            # But we've already handled p.
            # So we should divide rem by p as many times as possible?
            # No, that's inefficient.
            
            # Correct logic:
            # 1. Identify indices where rem is divisible by p.
            # 2. Update phi: phi[idx] = phi[idx] // p * (p-1)
            # 3. Update rem: while rem[idx] % p == 0: rem[idx] //= p
            
            # This while loop is hard to vectorize easily in NumPy for variable counts.
            # Alternative:
            # Since we are iterating p in increasing order, and p is prime:
            # If rem[idx] is divisible by p, we divide it by p.
            # It might still be divisible by p (i.e., p^2 divides n).
            # We need to divide it out completely to ensure that if a large prime q remains, it's not mixed with p.
            # But actually, if p^2 divides n, p is still a small prime.
            # The large prime factor check is for factors > LIMIT_TRIAL.
            # So we just need to remove all factors of p from rem.
            
            # Let's do a simple loop for the division if needed, or just one division and check?
            # If we only do one division, rem might still be divisible by p.
            # Then later, if we check rem > 1, we might think there's a large factor, but it's actually p again.
            # But we've already processed p.
            # So we MUST remove all factors of p from rem.
            
            # Vectorized division by p until not divisible is tricky.
            # However, for most numbers, p divides only once.
            # Let's just do:
            # rem[idx_start::p] //= p
            # Then check if any are still divisible by p?
            # If so, divide again?
            # This could be slow if many numbers have high powers of small primes.
            # But small primes are few.
            
            # Let's try a different approach for the residual part:
            # Just keep the original number and factorize the residual using Pollard Rho if > 1.
            # This avoids the complexity of segmented rem updates.
            
            # Re-evaluating:
            # The current phi_large is slow because of individual Pollard Rho.
            # Segmented sieve is faster if we can efficiently update rem.
            
            # Let's stick to the simple update:
            # phi_vals[idx_start::p] = phi_vals[idx_start::p] // p * (p-1)
            # rem[idx_start::p] //= p
            # If rem[idx_start::p] is still divisible by p, it's fine, we just leave it?
            # No, if we leave it, and later we check if rem > 1, we might factorize p again?
            # No, we don't check rem for small primes again.
            # We only check rem for factors > LIMIT_TRIAL.
            # So if rem still has p, it's a problem because p is NOT > LIMIT_TRIAL.
            # So we MUST remove all p's from rem.
            
            # Let's implement a loop that divides rem by p until it's not divisible.
            # We can do this iteratively for the slice.
            
            mask = rem[idx_start::p] % p == 0
            # This is always true for the first division.
            # We need to divide until no longer divisible.
            
            # Optimization: Most numbers are square-free with respect to small primes.
            # Let's just divide once, and then check if any remainders are still divisible.
            # If so, divide again.
            
            slice_rem = rem[idx_start::p]
            slice_phi = phi_vals[idx_start::p]
            
            # Update phi
            slice_phi = slice_phi // p * (p-1)
            
            # Update rem
            # Divide by p
            slice_rem = slice_rem // p
            
            # Check if still divisible
            while np.any(slice_rem % p == 0):
                mask = slice_rem % p == 0
                slice_rem[mask] //= p
            
            # Write back
            rem[idx_start::p] = slice_rem
            phi_vals[idx_start::p] = slice_phi

        # Now, for each i, if rem[i] > 1, it is a product of primes > LIMIT_TRIAL.
        # Since LIMIT_TRIAL = 10^5, and max number is 10^12, the residual can be:
        # 1. A prime > 10^5.
        # 2. A product of two primes > 10^5. (e.g., 10^6 * 10^6 = 10^12)
        # 3. A product of three? 10^5 * 10^5 * 10^5 = 10^15 > 10^12. So at most 2 large primes.
        
        # So we can check if rem[i] is prime. If yes, update phi.
        # If no, it's a product of two primes. We can find them by factoring rem[i] (which is fast since it's semiprime with large factors? No, Pollard Rho is good for semiprimes).
        
        for i in range(count):
            if rem[i] > 1:
                # rem[i] is either prime or product of two primes > LIMIT_TRIAL
                if is_prime_mr(rem[i]):
                    phi_vals[i] = phi_vals[i] // rem[i] * (rem[i] - 1)
                else:
                    # Factorize rem[i]
                    # It's a semiprime p*q with p,q > 10^5
                    # Use Pollard Rho
                    d = pollard_rho(rem[i])
                    p1 = d
                    p2 = rem[i] // d
                    phi_vals[i] = phi_vals[i] // p1 * (p1 - 1)
                    if p2 != p1:
                        phi_vals[i] = phi_vals[i] // p2 * (p2 - 1)
                    # Note: if p1 == p2, we divided by p1 once. But if rem[i] = p1^2, 
                    # then phi should be divided by p1 twice?
                    # No, phi(p^2) = p^2 - p = p(p-1).
                    # phi(n) = n * (1-1/p).
                    # If n = p^2, phi(n) = p^2 (1-1/p) = p(p-1).
                    # Our formula: phi_val = n * prod(1-1/p).
                    # We updated phi_val for small primes.
                    # Now for large primes.
                    # If rem[i] = p^2, then is_prime_mr(rem[i]) is False.
                    # pollard_rho returns p.
                    # p1 = p, p2 = p.
                    # phi_val = phi_val // p * (p-1).
                    # But we need to multiply by (1-1/p) for EACH distinct prime factor.
                    # The distinct prime factor is just p.
                    # So we only apply (1-1/p) once.
                    # This is correct.
        
        return phi_vals

    def max_distinct_run(vals, max_k=200):
        n = len(vals)
        best_k = 0
        best_idx = 0
        
        for i in range(n):
            if n - i < best_k:
                break
            seen = set()
            for j in range(i, min(i + max_k, n)):
                v = vals[j]
                if v in seen:
                    break
                seen.add(v)
                curr_k = j - i + 1
                if curr_k > best_k:
                    best_k = curr_k
                    best_idx = i
                    if best_k == max_k:
                        return best_k, best_idx
        return best_k, best_idx

    best_k = 1
    best_s = 1

    # --- Phase 1: Fast NumPy Sieve for Small Range ---
    # Use a larger range, e.g., 10^7
    N = 10 * 10**6
    t0 = now()
    
    try:
        phi_arr = np.arange(N + 1, dtype=np.int64)
        
        # Sieve of Eratosthenes for phi
        # Vectorized approach
        for i in range(2, N + 1):
            if phi_arr[i] == i: # i is prime
                phi_arr[i::i] -= phi_arr[i::i] // i
            
            if i > 10**6 and (i % 1000000 == 0):
                if now() > DEADLINE - 50.0:
                    break

        vals = phi_arr[1:]
        k, idx = max_distinct_run(vals, 200)
        if k > best_k:
            best_k = k
            best_s = idx + 1
            
    except Exception as e:
        # Fallback
        pass

    # --- Phase 2: Segmented Search in Larger Ranges ---
    # Search in chunks of 1000 numbers in the range [10^7, 10^12]
    # We can't scan all, so we use random sampling or heuristic.
    # Let's try scanning a few large chunks.
    
    if now() < DEADLINE - 20.0:
        # Try some specific large ranges
        # e.g., around 10^10, 10^11
        ranges_to_check = [
            10**7 + 1000,
            10**8 + 1000,
            10**9 + 1000,
            10**10 + 1000,
            10**11 + 1000,
            10**12 - 10000,
        ]
        
        # Add some random large ranges
        random.seed(42)
        for _ in range(10):
            s = random.randint(10**7, 10**12 - 1000)
            ranges_to_check.append(s)
        
        for s in ranges_to_check:
            if now() > DEADLINE - 5.0:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            
            try:
                # Check a window of 1000 numbers starting at s
                # But we only need to find a run of 200.
                # Let's compute phi for 200 numbers starting at s.
                # Actually, to be safe, compute for 300 numbers and find max run.
                vals = phi_batch_segmented(s, 300)
                k, idx = max_distinct_run(vals, 200)
                if k > best_k:
                    best_k = k
                    best_s = s + idx
                    if best_k == 200:
                        break
            except Exception:
                continue

    return {'start': best_s, 'length': best_k}
