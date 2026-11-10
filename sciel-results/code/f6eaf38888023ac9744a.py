import math
import numpy as np

def construct():
        N = 123457
        lpf = np.zeros(N + 1, dtype=np.int64)
        lpf[1] = 1
        
        is_prime = np.ones(N + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        
        for p in range(2, N + 1):
            if is_prime[p]:
                lpf[p::p] = p
        
        # Vectorized search
        # a = 1 to N-1
        a = np.arange(1, N)
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        max_p = np.maximum(pa, pb)
        min_idx = np.argmin(max_p)
        best_a = a[min_idx]
        best_b = b[min_idx]
        
        return [int(best_a), int(best_b)]
