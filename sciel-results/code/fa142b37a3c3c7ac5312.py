import numpy as np

def construct():
            N = 1000001
            # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
            spf = np.zeros(N + 1, dtype=np.int64)
            spf[1:] = np.arange(1, N + 1)
            
            # Sieve
            for i in range(2, int(np.sqrt(N)) + 1):
                if spf[i] == i:  # i is prime
                    # Mark multiples of i
                    # We only need to mark if spf[j] is still j
                    # Using NumPy slice assignment is faster, but we need to preserve existing smaller spf
                    # So we have to iterate or use a mask
                    # Mask approach:
                    mask = spf[i*i::i] == spf[i*i::i] # This is always true initially? No.
                    # Actually, spf[j] is j if not touched.
                    # So we want to update spf[j] = i where spf[j] == j.
                    
                    # Let's stick to the Python loop for clarity and safety, it's fast enough.
                    for j in range(i * i, N + 1, i):
                        if spf[j] == j:
                            spf[j] = i
            
            # 2. Compute Largest Prime Factor (LPF)
            lpf = np.zeros(N + 1, dtype=np.int64)
            lpf[1] = 1
            
            for i in range(2, N + 1):
                if spf[i] == i:
                    lpf[i] = i
                else:
                    lpf[i] = max(spf[i], lpf[i // spf[i]])
            
            # 3. Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
            start = 10000
            end = 1000000
            
            lpf_n = lpf[start:end + 1]
            lpf_n1 = lpf[start + 1:end + 2]
            
            metrics = np.maximum(lpf_n, lpf_n1)
            min_idx = np.argmin(metrics)
            best_n = start + min_idx
            
            return int(best_n)
