"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    N = 1000001
    powerful = np.ones(N + 1, dtype=np.int64)
    pp = np.ones(N + 1, dtype=np.int64)
    spf = np.zeros(N + 1, dtype=np.int32)
    primes = []
    
    # Base case
    powerful[1] = 1
    pp[1] = 1
    spf[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            primes.append(i)
            spf[i] = i
            pp[i] = i
            powerful[i] = 1  # Prime, exponent 1, powerful part 1
        
        for p in primes:
            if i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                # p divides i
                pp[i * p] = pp[i] * p
                rest = i // pp[i]
                powerful[i * p] = powerful[rest] * pp[i * p]
                break
            else:
                # p < spf[i], so p does not divide i
                pp[i * p] = p
                powerful[i * p] = powerful[i]
    
    # Find max product
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]
    
    # Vectorized scan for max
    # We want max of powerful[n] * powerful[n+1] for n in [10000, 999999]
    # Let A = powerful[10000:1000000]
    # Let B = powerful[10001:1000001]
    # Product = A * B
    # Index of max in Product gives offset from 10000
    
    A = powerful[10000:1000000]
    B = powerful[10001:1000001]
    products = A * B
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
