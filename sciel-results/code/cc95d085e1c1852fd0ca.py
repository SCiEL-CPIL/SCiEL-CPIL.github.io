import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi_batch(start, length):
        """Compute phi for all numbers in [start, start+length-1] efficiently."""
        if length <= 0:
            return []
        
        # Initialize arrays
        # remaining[i] will hold the unprocessed part of (start+i)
        remaining = [0] * length
        # phi_val[i] will hold the running product for phi
        phi_val = [0] * length
        
        for i in range(length):
            n = start + i
            remaining[i] = n
            phi_val[i] = n
            
        # Process small primes
        # We only need primes up to sqrt(max_n)
        max_n = start + length - 1
        limit_p = int(math.isqrt(max_n))
        
        for p in primes:
            if p > limit_p:
                break
                
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            
            # Iterate through multiples
            for multiple in range(first, start + length, p):
                idx = multiple - start
                # Check if p divides the remaining part
                if remaining[idx] % p == 0:
                    # Update phi: phi(n) = n * (1 - 1/p)
                    phi_val[idx] -= phi_val[idx] // p
                    # Remove all factors of p from remaining
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(length):
            if remaining[i] > 1:
                # remaining[i] is a prime factor
                phi_val[i] -= phi_val[i] // remaining[i]
                
        return phi_val

    def check_distinct(phi_vals):
        """Check if a list of values are pairwise distinct."""
        if len(phi_vals) <= 1:
            return True
        seen = set()
        for v in phi_vals:
            if v in seen:
                return False
            seen.add(v)
        return True

    best_start = 1
    best_len = 1
    
    # Target length is 200
    target_len = 200
    
    # Search strategy:
    # 1. Random sampling in high ranges
    # 2. If a run is found, try to extend it
    
    random.seed(42)
    
    # Define search ranges
    # High ranges are better because phi values are more spread out?
    # Actually, for large n, phi(n) is close to n, so distinctness is easier.
    ranges = [
        (10**11, 10**12 - 200),
        (10**10, 10**11 - 200),
        (10**9, 10**10 - 200)
    ]
    
    found_full = False
    
    # Phase 1: Random search for full length 200
    while time.time() < deadline - 20:
        # Pick a random range
        r_min, r_max = random.choice(ranges)
        if r_min >= r_max:
            continue
            
        s = random.randint(r_min, r_max)
        
        # Check window [s, s+199]
        phi_vals = phi_batch(s, target_len)
        
        if check_distinct(phi_vals):
            best_start = s
            best_len = target_len
            found_full = True
            break
            
    # Phase 2: If not found, search for shorter runs and try to extend
    if not found_full:
        # Try lengths 100, 50, 25
        for target_l in [100, 50, 25]:
            if time.time() > deadline - 10:
                break
                
            while time.time() < deadline - 5:
                r_min, r_max = random.choice(ranges)
                if r_min >= r_max:
                    continue
                    
                s = random.randint(r_min, r_max)
                
                phi_vals = phi_batch(s, target_l)
                
                if check_distinct(phi_vals):
                    if target_l > best_len:
                        best_len = target_l
                        best_start = s
                        
                        # Try to extend this run
                        # Extend to the left
                        while best_start > 1 and best_len < 200:
                            new_s = best_start - 1
                            # Check if phi(new_s) is distinct from existing
                            # We need to recompute or just check one new value
                            # Efficiently: compute phi(new_s) and check against set of existing phis
                            # But we don't have the set easily. Let's just recompute the whole window
                            # Or better, just check if phi(new_s) is not in the current set.
                            # To do this, we need the current set.
                            # Let's just recompute the window of size best_len + 1
                            pass # Simplified: just break and let the next iteration handle it or do a local search
                        
                        # Local search around best_start
                        # Try offsets -10 to 10
                        for offset in range(-10, 11):
                            new_s = best_start + offset
                            if new_s < 1:
                                continue
                            if new_s + 200 > 10**12:
                                continue
                            
                            # Try to find max length starting at new_s
                            # Binary search or linear increase?
                            # Let's just check 200
                            phi_vals_200 = phi_batch(new_s, 200)
                            if check_distinct(phi_vals_200):
                                best_start = new_s
                                best_len = 200
                                found_full = True
                                break
                            
                            # If not 200, check 100
                            if best_len < 100:
                                phi_vals_100 = phi_batch(new_s, 100)
                                if check_distinct(phi_vals_100):
                                    best_start = new_s
                                    best_len = 100
                        if found_full:
                            break
                            
            if found_full:
                break

    return {'start': best_start, 'length': best_len}
