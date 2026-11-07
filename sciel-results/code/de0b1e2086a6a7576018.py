import math
import time
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Increase N to 10^7
    N = 10000000
    
    # Use array module for memory efficiency
    # 'q' is signed long long (64-bit)
    # 'i' is signed int (32-bit)
    
    sigma = array('q', [0]) * (N + 1)
    ppow = array('q', [0]) * (N + 1)
    sigma_rest = array('q', [0]) * (N + 1)
    spf = array('i', [0]) * (N + 1)
    
    sigma[1] = 1
    ppow[1] = 1
    sigma_rest[1] = 1
    
    primes = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            ppow[i] = i
            sigma_rest[i] = 1
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                ppow[i * p] = ppow[i] * p
                sigma_rest[i * p] = sigma_rest[i]
                sigma[i * p] = sigma[i] + ppow[i] * p * sigma_rest[i]
                break
            else:
                ppow[i * p] = p
                sigma_rest[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
                
        if time.time() > deadline - 10.0:
            break

    # Find the best group
    # We need to map M -> list of k
    # M can be up to 10^18, so we use a dict
    groups = {}
    max_k_processed = 1
    
    # We only need to process k where sigma[k] != 0
    # Since we might break early, we check up to max_i
    # But the loop above breaks the outer loop if time runs out.
    # We need to track the last i processed.
    
    # Actually, the break is inside the outer loop.
    # If we break, we don't know exactly where.
    # Let's just iterate up to N and check if sigma[k] != 0.
    # But if we broke early, many sigma[k] are 0.
    
    # Let's restructure to track limit.
    limit = N
    # The break condition `if time.time() > deadline - 10.0: break` exits the outer loop.
    # So `i` stops at some value.
    # We can just iterate 1 to N and check `if sigma[k] == 0: continue`.
    
    for k in range(1, N + 1):
        s = sigma[k]
        if s == 0:
            continue
        M = k * s
        if M not in groups:
            groups[M] = []
        groups[M].append(k)
        
        if time.time() > deadline - 5.0:
            break

    # Find best group
    best_group = []
    best_c = 0
    for c, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_c = c
            
    if len(best_group) >= 3:
        M = int(best_c)
        # Verify k <= sqrt(M)
        sqrt_M = math.isqrt(M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': M, 'k': valid_ks}
            
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
