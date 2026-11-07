import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Primes to use
    primes = [2, 3, 5, 7, 11, 13]
    
    # Precompute prime powers and their contributions to M
    # p^e * sigma(p^e)
    # sigma(p^e) = (p^(e+1)-1)/(p-1)
    # term = p^e * (p^(e+1)-1)/(p-1)
    
    max_m = 10**18
    
    # We will generate k recursively
    # k = prod p_i^e_i
    # M = prod term_i
    
    # To avoid deep recursion or overhead, we can use iterative generation or a simple DFS
    # Given the small number of primes, DFS is fine.
    
    solutions = {} # M -> list of k's
    
    # Max exponents estimated to keep k small enough so M <= 10^18
    # If k is very large, M is very large.
    # k <= 10^9 is the hard limit for k.
    # But M = k * sigma(k) <= 10^18.
    # sigma(k) >= k, so k^2 <= 10^18 => k <= 10^9.
    # sigma(k) is usually > k.
    
    def gen(idx, current_k, current_m):
        if time.time() > deadline:
            return
        
        if idx == len(primes):
            if current_m <= max_m:
                if current_k in solutions.get(current_m, []):
                    return
                if current_m in solutions:
                    solutions[current_m].append(current_k)
                else:
                    solutions[current_m] = [current_k]
            return
        
        p = primes[idx]
        # Calculate max exponent for this prime such that current_k * p^e <= 10^9 (rough bound)
        # Actually, we can just iterate e and break if current_m * term > max_m
        # Or break if current_k * p^e > 10^9
        
        e = 0
        k_val = current_k
        m_val = current_m
        
        while True:
            # Check time
            if time.time() > deadline:
                return
            
            gen(idx + 1, k_val, m_val)
            
            # Prepare for next exponent
            # term = p^e * sigma(p^e)
            # We need to update m_val by multiplying by term for the next e?
            # No, m_val is the product of terms for primes 0..idx-1.
            # For prime p at idx, we add its contribution.
            
            # Let's restructure:
            # m_val currently contains contributions for primes < idx.
            # We need to multiply by (p^e * sigma(p^e)) for the current prime.
            
            # Calculate p^e * sigma(p^e)
            # sigma(p^e) = 1 + p + ... + p^e
            # term_e = p^e * (1 + p + ... + p^e)
            
            # It is more efficient to compute term_e iteratively.
            # term_0 = 1 * 1 = 1
            # term_{e+1} ?
            # Let's just compute from scratch or keep track of p^e and sigma_val.
            
            # Optimization:
            # Keep p_pow = p^e
            # Keep sig = 1 + p + ... + p^e
            # term = p_pow * sig
            
            # Start e=0:
            # p_pow = 1, sig = 1, term = 1
            # k_val = current_k * 1
            # m_val_new = current_m * 1
            
            # Next e:
            # p_pow *= p
            # sig += p_pow
            # term = p_pow * sig
            
            # We need to be careful with the loop structure.
            pass

    # Let's rewrite gen properly
    
    def gen_v2(idx, current_k, current_m):
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            if current_m <= max_m:
                if current_m in solutions:
                    solutions[current_m].append(current_k)
                else:
                    solutions[current_m] = [current_k]
            return
        
        p = primes[idx]
        p_pow = 1
        sig = 1
        
        # Limit k to 10^9
        # Limit M to 10^18
        while True:
            next_k = current_k * p_pow
            if next_k > 10**9:
                break
                
            term = p_pow * sig
            next_m = current_m * term
            if next_m > max_m:
                break
                
            gen_v2(idx + 1, next_k, next_m)
            
            if time.time() > deadline:
                return
                
            p_pow *= p
            sig += p_pow
            
    gen_v2(0, 1, 1)
    
    # Find best M
    best_m = 0
    best_ks = []
    max_count = 0
    
    for m, ks in solutions.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_m = m
            best_ks = ks
            
    if max_count == 0:
        # Fallback to seed
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_m, 'k': best_ks}
