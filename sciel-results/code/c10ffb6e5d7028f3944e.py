import time

def construct():
        start_time = time.time()
        deadline = start_time + 50.0 # Safe margin
        
        N = 1000000
        # SPF Sieve
        spf = list(range(N + 1))
        for i in range(2, int(N**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Compute Powerful Part P
        P = [1] * (N + 1)
        P[0] = 0
        for i in range(2, N + 1):
            p = spf[i]
            m = i // p
            if m % p == 0:
                k = i
                while k % p == 0:
                    k //= p
                P[i] = P[k] * (i // k)
            else:
                P[i] = P[m]
            
            if i % 100000 == 0:
                if time.time() > deadline:
                    break
        
        # Find Max
        max_val = 0
        best_n = 10000
        for n in range(10000, N):
            score = P[n] * P[n+1]
            if score > max_val:
                max_val = score
                best_n = n
                
        return best_n
