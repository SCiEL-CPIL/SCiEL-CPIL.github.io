import numpy as np

def construct():
    N = 1000001
    
    # We need arrays of size N+1
    # Using lists might be faster for small ints than numpy for element-wise updates in a tight loop?
    # Or numpy for the final scan.
    # Let's use Python lists for the sieve part as numpy scalar operations in loops are slow.
    
    spf = [0] * (N + 1)
    exp = [0] * (N + 1)
    powerful = [1] * (N + 1)
    primes = []
    
    powerful[1] = 1
    # spf[1] not strictly needed but good practice
    spf[1] = 1
    exp[1] = 0
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            exp[i] = 1
            powerful[i] = 1
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            
            j = i * p
            spf[j] = p
            
            if p < spf[i]:
                # p does not divide i
                exp[j] = 1
                powerful[j] = powerful[i]
            else:
                # p == spf[i], so p divides i
                exp[j] = exp[i] + 1
                if exp[i] == 1:
                    # Transition from p^1 to p^2
                    powerful[j] = powerful[i] * p * p
                else:
                    # Transition from p^k (k>=2) to p^(k+1)
                    powerful[j] = powerful[i] * p
                break
    
    # Find max product
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]
    
    # Use numpy for the scan to speed it up
    # Convert to numpy array
    pow_arr = np.array(powerful, dtype=np.int64)
    # We need n in [10000, 1000000]
    # products = pow_arr[10000:1000000] * pow_arr[10001:1000001]
    # This creates a large array, but 10^6 is fine.
    
    products = pow_arr[10000:1000000] * pow_arr[10001:1000001]
    idx = np.argmax(products)
    best_n = 10000 + int(idx)
    
    return best_n
