import time


def construct():
        start_time = time.time()
        # Deadline guard
        deadline = start_time + 85.0

        N_MAX = 1000000
        N_MIN = 10000
        limit = N_MAX + 1

        # 1. Linear Sieve for Smallest Prime Factor (SPF)
        spf = list(range(limit + 1))
        spf[0] = 0
        spf[1] = 1
        
        # Optimization: Only iterate up to sqrt(limit) for sieve initialization
        # Standard Linear Sieve is O(N)
        primes = []
        for i in range(2, limit + 1):
            if spf[i] == 0: # 0 means unvisited, which implies prime in this setup if we init to 0
                spf[i] = i
                primes.append(i)
            for p in primes:
                if p > spf[i] or i * p > limit:
                    break
                spf[i * p] = p
        # Note: The above is a standard linear sieve. 
        # The previous program used a simpler sieve which is also O(N log log N) or similar, but the linear one is strictly faster asymptotically.
        # However, the simple sieve in the prompt:
        # for i in range(2, int(max_val**0.5) + 1):
        #     if spf[i] == i:
        #         for j in range(i * i, max_val + 1, i):
        #             if spf[j] == j:
        #                 spf[j] = i
        # This is also very fast for 10^6. Let's use the simpler one as it has less overhead per iteration in Python (no inner loop over primes list).

        spf = list(range(limit + 1))
        for i in range(2, int(limit**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        # 2. Compute Powerful Part
        powerful_part = [1] * (limit + 1)
        
        for n in range(2, limit + 1):
            m = n
            pp = 1
            while m > 1:
                p = spf[m]
                exp = 0
                # Count exponent
                while m % p == 0:
                    m //= p
                    exp += 1
                if exp >= 2:
                    pp *= (p ** exp)
            powerful_part[n] = pp
            
            # Time check
            if n % 100000 == 0 and time.time() > deadline:
                break

        # 3. Find Max Product
        best_val = 0
        best_n = N_MIN
        
        for n in range(N_MIN, N_MAX + 1):
            val = powerful_part[n] * powerful_part[n + 1]
            if val > best_val:
                best_val = val
                best_n = n
                
        return best_n
