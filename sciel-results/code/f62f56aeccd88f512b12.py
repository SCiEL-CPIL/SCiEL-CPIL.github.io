import math
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit

    # We will search for k up to a certain limit N
    # sigma(k) can be computed via sieve
    # Let's start with N = 100000. If time allows, we can go higher.
    # But first, let's see if N=100000 yields a better result than 3.
    
    def find_best_solution(N):
        # Sieve for sigma(k)
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
        
        # Map M to list of k
        m_to_k = {}
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val > 10**18:
                continue
            if M_val not in m_to_k:
                m_to_k[M_val] = []
            m_to_k[M_val].append(k)
        
        best_M = 0
        best_ks = []
        for M_val, ks in m_to_k.items():
            # Check constraint k <= sqrt(M)
            valid_ks = [k for k in ks if k <= math.isqrt(M_val)]
            if len(valid_ks) > len(best_ks):
                best_ks = valid_ks
                best_M = M_val
        
        return best_M, best_ks

    # Try increasing N
    # N=10000 should be fast
    # N=100000 should be fast
    # N=1000000 might take a few seconds
    
    current_best_M, current_best_ks = 333312, [336, 372, 434] # Seed solution
    
    # Try N = 200000 first
    N = 200000
    if time.time() - start_time < deadline - 10:
        M, ks = find_best_solution(N)
        if len(ks) > len(current_best_ks):
            current_best_M, current_best_ks = M, ks
            
    # If we still only have 3, try larger N
    if len(current_best_ks) <= 3 and time.time() - start_time < deadline - 20:
        N = 500000
        M, ks = find_best_solution(N)
        if len(ks) > len(current_best_ks):
            current_best_M, current_best_ks = M, ks
            
    if len(current_best_ks) <= 3 and time.time() - start_time < deadline - 30:
        N = 1000000
        M, ks = find_best_solution(N)
        if len(ks) > len(current_best_ks):
            current_best_M, current_best_ks = M, ks
            
    return {'M': current_best_M, 'k': current_best_ks}
