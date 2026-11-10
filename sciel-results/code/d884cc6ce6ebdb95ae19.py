import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    best_n = 10000
    best_prod = 0
    
    # Sieve of Eratosthenes to find smallest prime factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to compute powerful part
    def powerful_part(x):
        pp = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                pp *= (p ** count)
        return pp
    
    # Search for max product
    for n in range(10000, N):
        if time.time() > deadline:
            break
        pn = powerful_part(n)
        if pn == 1:
            # Optimization: if P(n)=1, product is small (max P(n+1) <= 10^6)
            # But we still need to check if P(n+1) is large?
            # Max possible product with P(n)=1 is 1 * 10^6 = 10^6.
            # We already found 300M+. So we can skip if best_prod > 10^6.
            if best_prod > 1000000:
                continue
        pn1 = powerful_part(n + 1)
        prod = pn * pn1
        if prod > best_prod:
            best_prod = prod
            best_n = n
            
    return best_n
