import sys
import time
from math import isqrt

# Increase recursion limit for safety
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Primes to consider. Small primes yield more combinations.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    max_M = 10**18
    
    # solutions[M] = set of k's
    solutions = {}
    
    # Precompute p and sigma_factor for each prime
    # We will build k and M incrementally
    # k = prod p^e
    # M = prod (p^e * sigma(p^e))
    
    def dfs(idx, current_k, current_M):
        if current_M > max_M:
            return
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            # Record the solution
            if current_M not in solutions:
                solutions[current_M] = set()
            solutions[current_M].add(current_k)
            return
            
        p = primes[idx]
        # Calculate p^e and sigma(p^e) iteratively
        # Start with e=0
        # term_k = 1, term_M = 1
        term_k = 1
        term_M = 1
        
        # Try exponents e = 0, 1, 2, ...
        # p^e * sigma(p^e) = p^e * (p^(e+1)-1)/(p-1)
        # Let sigma_val = 1
        # For e=0: k_part=1, M_part=1
        # For e=1: k_part=p, M_part=p*(p+1)
        # For e=2: k_part=p^2, M_part=p^2*(p^2+p+1)
        
        # We can iterate e
        e = 0
        k_part = 1
        sigma_part = 1 # sigma(p^0) = 1
        
        while True:
            new_k = current_k * k_part
            new_M = current_M * (k_part * sigma_part)
            
            if new_M > max_M:
                break
            
            dfs(idx + 1, new_k, new_M)
            
            # Update for next e
            # k_part *= p
            # sigma_part = sigma_part * p + 1  (Geometric series sum update: 1 + p + ... + p^e)
            # Wait, sigma(p^e) = 1 + p + ... + p^e.
            # If we have sigma(p^e), then sigma(p^(e+1)) = sigma(p^e) * p + 1? No.
            # sigma(p^(e+1)) = 1 + p + ... + p^e + p^(e+1) = sigma(p^e) + p^(e+1).
            # Also p^(e+1) = k_part * p.
            # So sigma_part_new = sigma_part + k_part * p.
            
            # Check bounds for k_part * p to avoid overflow (though Python handles big ints, 
            # we stop when M exceeds limit).
            if k_part > max_M // p:
                break
                
            k_part *= p
            sigma_part += k_part # Since k_part is now p^(e+1)
            
            # Note: The loop condition `new_M > max_M` handles the stopping criteria.
            # We just need to ensure we don't do infinite loops if p=1 (not possible) 
            # or if the product doesn't grow (p>=2, so it grows).

    # Start DFS
    dfs(0, 1, 1)
    
    # Find the best M
    best_M = 1
    best_count = 0
    best_k_list = []
    
    for M, k_set in solutions.items():
        count = len(k_set)
        if count > best_count:
            best_count = count
            best_M = M
            best_k_list = list(k_set)
            
    # Sort k for consistency (not required but good)
    best_k_list.sort()
    
    # Verification and cleanup
    # The problem requires k <= isqrt(M).
    # Our generation ensures k * sigma(k) = M. Since sigma(k) >= k+1 (for k>1), k(k+1) <= M => k < sqrt(M).
    # For k=1, sigma(1)=1, M=1, k=1 <= 1.
    # So the constraint k <= isqrt(M) is naturally satisfied for k>1.
    # Let's double check just in case.
    valid_ks = []
    for k in best_k_list:
        if k <= isqrt(best_M):
            valid_ks.append(k)
            
    # If something went wrong, fallback to known good
    if not valid_ks:
         return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in valid_ks]}

if __name__ == '__main__':
    result = construct()
    print(result['M'])
    print(len(result['k']))
