"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    
    N = 1000000  # search k up to 1e6
    
    # Compute sigma(k) for all k up to N using sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Count occurrences of M = k * sigma(k)
    from collections import defaultdict
    count = defaultdict(list)
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            count[M_val].append(k)
    
    # Find M with most solutions where k_i <= sqrt(M)
    best_M = None
    best_k = []
    for M_val, ks in count.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_k):
            best_k = valid_ks
            best_M = M_val
    
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify and return
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
