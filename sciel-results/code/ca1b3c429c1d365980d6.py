"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: Use NumPy for fast sigma sieve and grouping.
    # K=10,000,000 is a safe balance of time and memory.
    K = 10000000
    
    try:
        # 1. Compute Sigma using NumPy
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
        
        # 2. Group by M
        groups = defaultdict(list)
        g = groups
        for k in range(1, K + 1):
            M = k * sigma[k]
            if M > 10**18:
                continue
            # k <= isqrt(M) is always true for k>=1
            g[M].append(k)
            
        # 3. Find best
        for M, ks in g.items():
            cnt = len(ks)
            if cnt > best_count:
                best_count = cnt
                best_M = M
                best_k = ks
                
    except MemoryError:
        # Fallback to smaller K if memory fails
        K = 2000000
        try:
            sigma = np.zeros(K + 1, dtype=np.int64)
            for d in range(1, K + 1):
                sigma[d::d] += d
            
            groups = defaultdict(list)
            g = groups
            for k in range(1, K + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    g[M].append(k)
            
            for M, ks in g.items():
                cnt = len(ks)
                if cnt > best_count:
                    best_count = cnt
                    best_M = M
                    best_k = ks
        except MemoryError:
            # Final fallback to pure Python small K
            K = 100000
            sigma = [0] * (K + 1)
            for d in range(1, K + 1):
                for m in range(d, K + 1, d):
                    sigma[m] += d
            
            groups = defaultdict(list)
            for k in range(1, K + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    groups[M].append(k)
            
            for M, ks in groups.items():
                cnt = len(ks)
                if cnt > best_count:
                    best_count = cnt
                    best_M = M
                    best_k = ks

    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
