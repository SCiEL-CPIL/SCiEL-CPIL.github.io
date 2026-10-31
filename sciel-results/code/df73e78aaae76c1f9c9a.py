import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Generate weight-4 vectors
    # Represented as (pos_mask, neg_mask)
    # pos_mask, neg_mask are 11-bit ints
    
    vecs4 = []
    for support in itertools.combinations(range(D), 4):
        # 16 sign combinations
        for mask in range(16):
            pos = 0
            neg = 0
            for i, idx in enumerate(support):
                if mask & (1 << i):
                    pos |= (1 << idx)
                else:
                    neg |= (1 << idx)
            vecs4.append((pos, neg))
            
    # Generate weight-2 vectors
    vecs2 = []
    for support in itertools.combinations(range(D), 2):
        for mask in range(4):
            pos = 0
            neg = 0
            for i, idx in enumerate(support):
                if mask & (1 << i):
                    pos |= (1 << idx)
                else:
                    neg |= (1 << idx)
            vecs2.append((pos, neg))
            
    # Helper: dot product
    def dot4(pos1, neg1, pos2, neg2):
        # u.v = |p1&p2| - |p1&n2| - |n1&p2| + |n1&n2|
        return (pos1 & pos2).bit_count() - (pos1 & neg2).bit_count() - (neg1 & pos2).bit_count() + (neg1 & neg2).bit_count()
        
    def dot2(pos1, neg1, pos2, neg2):
        return (pos1 & pos2).bit_count() - (pos1 & neg2).bit_count() - (neg1 & pos2).bit_count() + (neg1 & neg2).bit_count()

    # Greedy Independent Set for vecs4
    # Calculate degrees
    degrees = [0] * len(vecs4)
    for i in range(len(vecs4)):
        p1, n1 = vecs4[i]
        for j in range(i+1, len(vecs4)):
            p2, n2 = vecs4[j]
            if dot4(p1, n1, p2, n2) > 2:
                degrees[i] += 1
                degrees[j] += 1
                
    # Sort by degree
    sorted_indices = sorted(range(len(vecs4)), key=lambda x: degrees[x])
    
    selected4 = []
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        p, n = vecs4[idx]
        ok = True
        for p_s, n_s in selected4:
            if dot4(p, n, p_s, n_s) > 2:
                ok = False
                break
        if ok:
            selected4.append((p, n))
            
    # Add weight-2s
    selected2 = []
    for p, n in vecs2:
        if time.time() > deadline:
            break
        ok4 = True
        for p_s, n_s in selected4:
            # w . u <= 1
            if dot2(p, n, p_s, n_s) > 1:
                ok4 = False
                break
        if not ok4:
            continue
            
        ok2 = True
        for p_s, n_s in selected2:
            # w . v <= 0
            if dot2(p, n, p_s, n_s) > 0:
                ok2 = False
                break
        if ok2:
            selected2.append((p, n))
            
    # Construct output array
    # We need to convert (pos, neg) back to vector
    total_k = len(selected4) + len(selected2)
    result = np.zeros((total_k, D), dtype=np.int64)
    
    k = 0
    for p, n in selected4:
        for i in range(D):
            if p & (1 << i):
                result[k, i] = 1
            elif n & (1 << i):
                result[k, i] = -1
        k += 1
        
    for p, n in selected2:
        for i in range(D):
            if p & (1 << i):
                result[k, i] = 1
            elif n & (1 << i):
                result[k, i] = -1
        k += 1
        
    return result
