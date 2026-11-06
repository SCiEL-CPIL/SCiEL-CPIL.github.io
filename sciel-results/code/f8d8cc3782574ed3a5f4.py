import time

_START_TIME = time.time()

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def priority(vector, n):
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    h = _hash_vec(vector)
    
    # Penalize zeros to prioritize the {1,2}^n cap set core
    # Penalize imbalance between 1s and 2s to reduce risk of 1+1+1 or 2+2+2 triples
    imbalance = abs(c1 - c2)
    
    return h - 50.0 * c0 - 10.0 * imbalance
