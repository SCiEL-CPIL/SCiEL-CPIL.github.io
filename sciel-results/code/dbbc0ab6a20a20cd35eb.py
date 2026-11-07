import math
import numpy as np
from math import isqrt

def construct():
    K_max = 10**7
    # Sieve for sigma
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    for i in range(1, K_max + 1):
        sigma[i::i] += i
    
    counts = {}
    first_k = {}
    lists = {}
    
    max_M = 10**18
    
    for k in range(1, K_max + 1):
        M = k * sigma[k]
        if M > max_M:
            continue
        if M not in counts:
            counts[M] = 1
            first_k[M] = k
        else:
            counts[M] += 1
            if counts[M] == 2:
                lists[M] = [first_k[M], k]
            else:
                lists[M].append(k)
                
    best_M = 1
    best_k = [1]
    best_count = 1
    
    for M, ks in lists.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            
    # If no collisions found, best_count is 1. 
    # We should still return a valid M and k.
    # If best_count == 1, we can just pick any M from counts with count 1.
    # But the loop above only populates lists for count >= 2.
    # So if lists is empty, best_M remains 1, best_k [1].
    # Let's verify k=1 is valid. 1*sigma(1)=1. Yes.
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
