import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # We need to compute powerful_part for n and n+1.
    # If n can be 1000000, n+1 can be 1000001.
    # So we need SPF and pp up to 1000001.
    MAX_N = 1000001
    
    # Precompute smallest prime factor (SPF) for efficient factorization
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    def powerful_part(n):
        if n <= 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                result *= (p ** count)
        return result
    
    # Precompute powerful parts for all numbers
    pp = [0] * (MAX_N + 1)
    pp[0] = 1
    pp[1] = 1
    for i in range(2, MAX_N + 1):
        pp[i] = powerful_part(i)
        
        # Check time periodically during precomputation? 
        # Precomputation is fast, so probably not needed, but safe.
        if i % 100000 == 0 and time.time() > deadline:
            # If we run out of time, we can't compute pp for higher numbers.
            # But we need pp up to 1000001.
            # If we fail, we might return a suboptimal result or raise an error.
            # Given N=1e6, this should not happen.
            pass
    
    # Find the best n in [10000, 1000000]
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, MAX_N):  # MAX_N is 1000001, so range goes up to 1000000
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
