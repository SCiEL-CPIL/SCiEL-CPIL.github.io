import numpy as np
import itertools
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 580
    d = 11
    
    # Start with all weight-2 vectors: ±e_i ± e_j for i < j
    # Norm² = 2, conflict if dot >= 1 (i.e., they share a coordinate with same sign)
    # This gives C(11,2)*4 = 220 vectors
    # But we need min_dist² >= max_norm² = 2
    # dist² = |a-b|² = |a|² + |b|² - 2*a·b = 4 - 2*a·b
    # Need 4 - 2*a·b >= 2 => a·b <= 1
    # So conflict if a·b >= 2, which means they share both coordinates with same sign
    # Actually for weight-2: a·b can be -2,-1,0,1,2
    # a·b = 2 only if a == b, a·b = -2 only if a == -b
    # For distinct pairs with a≠±b: a·b ∈ {-1, 0, 1}
    # So all distinct weight-2 vectors (excluding ± pairs) have dist² >= 2
    # But we can't include both v and -v since dist(v,-v)² = 4*norm² = 8 >= 2 ✓
    # Actually dist²(v, -v) = |2v|² = 4*2 = 8 >= 2 ✓
    # So ALL 220 weight-2 vectors are mutually compatible!
    
    weight2 = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                weight2.append(v)
    
    points_arr = np.array(weight2, dtype=np.int64)
    best_k = len(points_arr)
    
    # Try to add weight-4 vectors: ±e_i±e_j±e_k±e_l
    # Norm² = 4. Need dist² >= 4 for all pairs.
    # For two weight-4 vectors a,b: dist² = 8 - 2*a·b >= 4 => a·b <= 2
    # a·b can range from -4 to 4. Conflict if a·b >= 3.
    # a·b = 3 means they share 3 coordinates with same sign and 1 with opposite, etc.
    # Actually max |a·b| = 4 when a=b or a=-b.
    # Conflict if a·b >= 3 or a·b <= -3 (since dist² = 8-2*dot, need >= 4 => dot <= 2)
    # Wait: dist² = |a|²+|b|²-2a·b = 4+4-2a·b = 8-2a·b
    # Need 8-2a·b >= 4 => a·b <= 2
    # So conflict if a·b >= 3
    
    # Also need dist between weight-2 and weight-4 >= max(2,4) = 4
    # dist²(w2, w4) = 2 + 4 - 2*w2·w4 = 6 - 2*w2·w4 >= 4 => w2·w4 <= 1
    # w2·w4 can be -2,-1,0,1,2. Conflict if w2·w4 >= 2.
    
    # This is complex. Let's just try greedy on weight-4 alone first.
    if time.time() < deadline:
        weight4 = []
        for pos in combinations(range(d), 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for p, s in zip(pos, signs):
                    v[p] = s
                weight4.append(v)
        
        weight4 = np.array(weight4, dtype=np.int64)
        n4 = len(weight4)
        
        # Conflict matrix for weight-4: conflict if dot >= 3
        conflict4 = np.zeros((n4, n4), dtype=bool)
        for i in range(n4):
            for j in range(i + 1, n4):
                if np.dot(weight4[i], weight4[j]) >= 3:
                    conflict4[i, j] = True
                    conflict4[j, i] = True
        
        # Greedy MIS on weight-4
        best_set = []
        for attempt in range(200):
            if time.time() >= deadline:
                break
            avail = np.ones(n4, dtype=bool)
            curr = []
            order = np.random.permutation(n4)
            for idx in order:
                if not avail[idx]:
                    continue
                curr.append(idx)
                avail[conflict4[idx]] = False
                avail[idx] = False
            if len(curr) > len(best_set):
                best_set = curr
                if len(best_set) > best_k:
                    best_k = len(best_set)
        
        if len(best_set) > best_k:
            points_arr = weight4[best_set]
    
    # Try mixed: keep all weight-2, greedily add compatible weight-4
    if time.time() < deadline:
        w2 = np.array(weight2, dtype=np.int64)
        w4_list = weight4.tolist()
        n2 = len(w2)
        
        # For each weight-4, check compatibility with all weight-2
        # w2·w4 <= 1 for all w2
        # Precompute: for a weight-4 vector, what's the max dot with any weight-2?
        # A weight-2 vector has ±1 in two positions.
        # dot(w2, w4) = ±w4[a] ± w4[b] for positions a,b of w2.
        # Max over all choices = sum of |w4[i]| for the two positions with largest |w4|
        # Since w4 has four ±1 entries, max dot = 2 (pick two positions where w4=±1 with matching signs)
        # So we need: for each weight-4, no weight-2 exists with dot >= 2.
        # But weight-2 includes all ±e_i±e_j, so for any weight-4 with at least two nonzero coords,
        # there exists a weight-2 with dot = 2. So NO weight-4 is compatible with all weight-2!
        # Unless we drop some weight-2 vectors.
        
        # Alternative: use only weight-4 vectors. Already tried above.
        
        # Try weight-3 vectors: ±e_i±e_j±e_k, norm²=3
        # dist² between two weight-3 = 6-2*dot >= 3 => dot <= 1.5 => dot <= 1
        # Conflict if dot >= 2
        # Cross with weight-2: dist² = 2+3-2*dot = 5-2*dot >= 3 => dot <= 1
        # w2·w3 can be -1,0,1. So always compatible with weight-2!
        # Cross with weight-4: dist² = 3+4-2*dot = 7-2*dot >= 4 => dot <= 1.5 => dot <= 1
        # w3·w4 can be -2,-1,0,1,2. Conflict if dot >= 2.
        
        # Strategy: take all weight-2 (220) + greedily add compatible weight-3
        w3 = []
        for pos in combinations(range(d), 3):
            for signs in product([1, -1], repeat=3):
                v = np.zeros(d, dtype=np.int64)
                for p, s in zip(pos, signs):
                    v[p] = s
                w3.append(v)
        w3 = np.array(w3, dtype=np.int64)
        n3 = len(w3)
        
        # Check which weight-3 are compatible with all weight-2
        # As shown above, all weight-3 are compatible with all weight-2.
        # Now check mutual compatibility among weight-3: conflict if dot >= 2
        conflict3 = np.zeros((n3, n3), dtype=bool)
        for i in range(n3):
            for j in range(i + 1, n3):
                if np.dot(w3[i], w3[j]) >= 2:
                    conflict3[i, j] = True
                    conflict3[j, i] = True
        
        # Greedy MIS on weight-3
        best_w3_set = []
        for attempt in range(300):
            if time.time() >= deadline:
                break
            avail = np.ones(n3, dtype=bool)
            curr = []
            order = np.random.permutation(n3)
            for idx in order:
                if not avail[idx]:
                    continue
                curr.append(idx)
                avail[conflict3[idx]] = False
                avail[idx] = False
            if len(curr) > len(best_w3_set):
                best_w3_set = curr
        
        # Combine all weight-2 + best weight-3 set
        if len(best_w3_set) > 0:
            combined = np.vstack([w2, w3[best_w3_set]])
            if len(combined) > len(points_arr):
                points_arr = combined
                best_k = len(combined)
    
    points_arr = np.around(points_arr).astype(np.int64)
    return points_arr
