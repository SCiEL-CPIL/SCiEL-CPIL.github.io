import math
import time
from collections import defaultdict

def _sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    # factor out 2
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    # odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Search for better solutions
    # We can search up to a certain limit
    # g(k) grows roughly as k^2 * (log log k)
    # For k = 10^5, g(k) ~ 10^10 * 3 = 3*10^10 < 10^18
    # For k = 10^6, g(k) ~ 10^12 * 3 = 3*10^12 < 10^18
    # For k = 10^7, g(k) ~ 10^14 * 3 = 3*10^14 < 10^18
    # For k = 10^8, g(k) ~ 10^16 * 3 = 3*10^16 < 10^18
    # For k = 10^9, g(k) ~ 10^18 * 3 = 3*10^18 > 10^18
    # So we can search up to ~5*10^8 if we are careful, but sigma computation is slow.
    # Let's try up to 200,000 first, then increase if time permits.
    
    N = 200000
    groups = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M > 10**18:
            # Since g(k) is generally increasing, we could break, but it's not strictly monotonic.
            # However, for large k, it will exceed.
            # We just skip.
            continue
        groups[M].append(k)
        
        # Update best if this group is larger
        if len(groups[M]) > best_count:
            # Check constraint k <= sqrt(M) is always true as discussed
            best_count = len(groups[M])
            best_M = M
            best_k = groups[M][:]
            
    return {'M': best_M, 'k': best_k}
