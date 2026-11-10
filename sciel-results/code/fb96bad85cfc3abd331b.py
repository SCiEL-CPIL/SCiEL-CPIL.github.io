import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # 1. Generate 3-smooth numbers u = 2^a * 3^b
    # and their sigma values A_u = u * sigma(u)
    # We limit u to a range such that u * v <= 10^18 and M <= 10^18 is possible.
    # Actually, we just need to generate enough candidates.
    
    def gen_smooth(primes, limit):
        """Generate numbers composed of primes in 'primes' up to 'limit'.
        Returns list of (num, sigma_num)."""
        results = [(1, 1)] # (1, sigma(1)=1)
        # Use a stack or recursion to generate combinations
        # Since the number of such numbers is small for small prime sets and moderate limits,
        # a simple recursive generator or iterative expansion works.
        
        # Let's do it iteratively to be safe and simple.
        # Start with 1
        current = [1]
        sigmas = [1]
        
        # This simple approach might miss combinations if not careful.
        # Better: recursive generation.
        
        def recurse(idx, current_num, current_sigma):
            if current_num > limit:
                return
            # Add current state
            results.append((current_num, current_sigma))
            
            if idx == len(primes):
                return
            
            p = primes[idx]
            # Multiply by p^k
            p_power = 1
            p_sigma_part = 1
            p_sum = 1
            while current_num * p_power <= limit:
                # Next power
                p_power *= p
                p_sum += p_power
                # sigma(p^e) = 1 + p + ... + p^e
                # We need to update sigma correctly.
                # sigma(current_num * p^e) = sigma(current_num) * sigma(p^e)
                # Wait, this recursive logic is slightly tricky with sigma.
                
                # Let's restart with a cleaner generation method.
                pass
        
        # Simpler method: just iterate all combinations of exponents.
        # For 2, 3: max a is 20 (2^20 ~ 1e6), max b is 13 (3^13 ~ 1.5e6).
        # For 5, 7, 11: max c is 9 (5^9 ~ 1.9e6), etc.
        
        return [] # Placeholder, will replace with actual implementation

    # Let's implement the generation properly inside construct to avoid global state issues or complex imports.
    
    def generate_smooth_numbers(primes, limit):
        """
        Generates numbers n = p1^a1 * p2^a2 * ... such that n <= limit.
        Returns a list of tuples (n, sigma(n)).
        """
        results = []
        num_primes = len(primes)
        
        # Max exponents
        max_exps = []
        for p in primes:
            e = 0
            prod = 1
            while prod * p <= limit:
                prod *= p
                e += 1
            max_exps.append(e)
        
        # Recursive helper
        def _gen(idx, current_n, current_sigma):
            if idx == num_primes:
                results.append((current_n, current_sigma))
                return
            
            p = primes[idx]
            max_e = max_exps[idx]
            
            # Try exponents 0 to max_e
            p_power = 1
            p_sigma_part = 1
            for e in range(max_e + 1):
                # If e > 0, we multiply by p
                # sigma(p^e) = 1 + p + ... + p^e
                # We accumulate p_sigma_part
                if e > 0:
                    p_power *= p
                    p_sigma_part += p_power
                
                next_n = current_n * p_power
                if next_n > limit:
                    break
                    
                next_sigma = current_sigma * p_sigma_part
                _gen(idx + 1, next_n, next_sigma)
                
        _gen(0, 1, 1)
        return results

    # Define prime sets
    # Set U: primes 2, 3
    # Set V: primes 5, 7, 11
    # Limits: We want k = u*v to be reasonably large to find collisions, 
    # but not so large that the list sizes explode.
    # Let's aim for u, v up to 10^6 or 10^7.
    # Number of 3-smooth up to 10^7 is ~200.
    # Number of 5,7,11-smooth up to 10^7 is ~500-1000.
    # Product size ~ 200 * 1000 = 200,000 pairs. This is very fast.
    
    limit_u = 10**7
    limit_v = 10**7
    
    # Generate U list
    u_data = generate_smooth_numbers([2, 3], limit_u)
    # u_data contains (u, sigma(u)). We need A_u = u * sigma(u).
    A_vals = [(u, u * su) for u, su in u_data]
    
    # Generate V list
    v_data = generate_smooth_numbers([5, 7, 11], limit_v)
    # v_data contains (v, sigma(v)). We need B_v = v * sigma(v).
    B_vals = [(v, v * sv) for v, sv in v_data]
    
    # Sort B_vals by B value? No, we just need to iterate.
    # To find collisions in M = A_u * B_v, we can iterate all pairs.
    # However, we want to find M with many solutions.
    # We will use a dictionary: m_to_ks = {M: [k1, k2, ...]}
    
    m_to_ks = defaultdict(set)
    
    # Optimization: Pre-filter B_vals?
    # We know M = A_u * B_v <= 10^18.
    # So B_v <= 10^18 / A_u.
    
    # Iterate U
    for u, A_u in A_vals:
        if time.time() > deadline - 5:
            break
        max_B = 10**18 // A_u
        # Iterate V
        for v, B_v in B_vals:
            if B_v > max_B:
                # Since B_vals is not sorted, we can't break easily unless we sort it.
                # Let's sort B_vals by B_v to allow breaking.
                continue
            
            M_val = A_u * B_v
            if M_val > 10**18:
                continue
                
            k = u * v
            # Check k <= sqrt(M_val)
            # M_val = k * sigma(k). sigma(k) >= k+1 for k>1.
            # So M_val > k^2. Thus k < sqrt(M_val).
            # For k=1, M=1, sqrt(1)=1, 1<=1.
            # So the condition k <= isqrt(M_val) is always satisfied for valid k>0.
            # But we should check to be safe.
            if k <= math.isqrt(M_val):
                m_to_ks[M_val].add(k)
                
    # Sort B_vals by B_v to allow breaking in inner loop?
    # The current loop continues if B_v > max_B. If we sort B_vals, we can break.
    # Let's refactor to sort B_vals first.
    
    # Re-do with sorted B_vals for efficiency
    B_vals_sorted = sorted(B_vals, key=lambda x: x[1])
    m_to_ks = defaultdict(set)
    
    for u, A_u in A_vals:
        if time.time() > deadline - 5:
            break
        max_B = 10**18 // A_u
        for v, B_v in B_vals_sorted:
            if B_v > max_B:
                break
            M_val = A_u * B_v
            k = u * v
            if k <= math.isqrt(M_val):
                m_to_ks[M_val].add(k)
                
    # Find the M with the most distinct k's
    best_M = 0
    best_ks = []
    max_count = 0
    
    for M_val, ks in m_to_ks.items():
        count = len(ks)
        if count > max_count:
            max_count = count
            best_M = M_val
            best_ks = list(ks)
            
    if best_M == 0:
        # Fallback
        return {'M': 1, 'k': [1]}
        
    # Final check and cleanup
    best_ks = sorted(set(int(k) for k in best_ks))
    sqrt_M = math.isqrt(best_M)
    best_ks = [k for k in best_ks if k <= sqrt_M]
    
    return {'M': int(best_M), 'k': best_ks}
