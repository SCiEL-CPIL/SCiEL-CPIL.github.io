import sys
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 100  # 100 seconds safety margin
    
    # Choose N. 10^7 is a good balance.
    N = 10**7
    
    # Linear sieve for sigma(n)
    # sigma[i] = sum of divisors of i
    # p_pow[i] = highest power of least prime factor of i
    # sigma_p[i] = sigma(p_pow[i])
    
    sigma = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    sigma_p = [0] * (N + 1)
    is_comp = [False] * (N + 1)
    primes = []
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_p[1] = 1
    
    for i in range(2, N + 1):
        if not is_comp[i]:
            primes.append(i)
            p_pow[i] = i
            sigma_p[i] = i + 1
            sigma[i] = i + 1
        
        for p in primes:
            val = i * p
            if val > N:
                break
            is_comp[val] = True
            
            if i % p == 0:
                p_pow[val] = p_pow[i] * p
                sigma_p[val] = sigma_p[i] + p_pow[val]
                sigma[val] = sigma[i] // sigma_p[i] * sigma_p[val]
                break
            else:
                p_pow[val] = p
                sigma_p[val] = p + 1
                sigma[val] = sigma[i] * (p + 1)
        
        if time.time() - start_time > deadline - 10:
            # If we are running out of time, we might need to break, 
            # but the sieve is O(N), so it should finish quickly.
            # If it doesn't, we return a fallback.
            pass

    # Check time
    if time.time() - start_time > deadline - 5:
        # Fallback to a known good solution if sieve was too slow
        return {'M': 333312, 'k': [336, 372, 434]}

    # Compute g(k) = k * sigma(k)
    # We only care about M <= 10^18
    MAX_M = 10**18
    g = [0] * (N + 1)
    valid_ks = []
    
    for i in range(1, N + 1):
        val = i * sigma[i]
        if val <= MAX_M:
            g[i] = val
            valid_ks.append(i)
    
    if not valid_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Count frequencies of g(k)
    # Filter out 0s (which are for indices where val > MAX_M, though g[i] stays 0)
    # But we only appended valid i to valid_ks? No, we need to count g[i] for valid i.
    # Let's create a list of g values for valid i.
    g_vals = [g[i] for i in valid_ks]
    
    counts = Counter(g_vals)
    
    if not counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    M, count = counts.most_common(1)[0]
    
    # Find all k such that g(k) == M
    ks = [i for i in valid_ks if g[i] == M]
    
    # Verify constraints
    # M <= 10^18 is guaranteed by construction
    # k <= sqrt(M) is guaranteed by k <= sigma(k)
    # k distinct is guaranteed by list comprehension over unique indices
    
    # Ensure types are correct
    M = int(M)
    ks = [int(k) for k in ks]
    
    return {'M': M, 'k': ks}

if __name__ == '__main__':
    res = construct()
    print(res)
