import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes to consider for smooth numbers
    primes = [2, 3, 5, 7, 11, 13]
    max_k = 10**9 # Since k <= sqrt(M) and M <= 10^18, k <= 10^9
    
    # Dictionary to store M -> list of k
    m_to_ks = defaultdict(list)
    
    # Recursive generation of smooth numbers
    # idx: index of current prime
    # current_k: the product so far
    # current_sigma: sigma(current_k)
    
    def gen_smooth(idx, current_k, current_sigma):
        if time.time() > deadline:
            return
        
        if idx == len(primes):
            if current_k > 1: # k=1 is trivial, sigma(1)=1, M=1
                M = current_k * current_sigma
                if M <= 10**18:
                    m_to_ks[M].append(current_k)
            return
        
        p = primes[idx]
        # Multiply by p^a
        # We need to update sigma correctly.
        # sigma(p^a) = 1 + p + ... + p^a
        # If current_k is coprime to p (which it is, as we process primes in order),
        # sigma(current_k * p^a) = sigma(current_k) * sigma(p^a)
        
        power = 1
        # Start with a=0
        # a=0: factor 1, sigma_factor 1
        gen_smooth(idx + 1, current_k, current_sigma)
        
        sigma_factor = 1
        # a=1, 2, ...
        while current_k * power <= max_k:
            power *= p
            # Update sigma_factor: sigma(p^a) = sigma(p^{a-1}) + p^a
            # Actually, easier to just compute sum.
            # But we can do: sigma_factor_new = sigma_factor_old + power
            # Wait, sigma(p^a) = 1 + p + ... + p^a.
            # If we have sigma(p^{a-1}), then sigma(p^a) = sigma(p^{a-1}) + p^a.
            sigma_factor += power
            gen_smooth(idx + 1, current_k * power, current_sigma * sigma_factor)
            
    gen_smooth(0, 1, 1)
    
    # Find the M with the most solutions
    best_M = 1
    best_ks = [1]
    
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            
    # Verify best_ks
    # The task requires distinct k_i. The generation ensures distinct k.
    # The task requires k_i <= isqrt(M).
    # We argued this is always true for k > 1.
    # For k=1, M=1, isqrt(1)=1, 1<=1.
    
    # Ensure types
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}
