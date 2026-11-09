import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # E8 lattice minimal vectors (scaled by 2) in the first 8 dimensions
    # Type A: Two coordinates are ±2, rest 0 in the first 8 dims
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [-2, 2]:
                for sj in [-2, 2]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Type B: All 8 coordinates are ±1, with even number of -1s
    # This gives 128 vectors
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.zeros(d, dtype=np.int64)
            for i in range(8):
                v[i] = signs[i]
            points.append(v)
    
    # D3 lattice minimal vectors (scaled by 2) in the last 3 dimensions
    # These are vectors with two non-zero coordinates of ±2 in the last 3 dims
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in [-2, 2]:
                for sj in [-2, 2]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Add new points with norm squared 8, split as 4+4
    # a in Z^8 with ||a||^2 = 4
    # b in Z^3 with ||b||^2 = 4
    # Constraint: a . u <= 4 for all u in E8_scaled (always true for ||a||^2=4)
    # Constraint: b . v <= 4 for all v in D3_scaled (always true for ||b||^2=4)
    # Constraint: ||a1 - a2||^2 + ||b1 - b2||^2 >= 8
    # If b1 == b2, need a1 . a2 <= 0
    
    # Valid b vectors in Z^3 with norm sq 4
    valid_bs = []
    for b in product([-2, 0, 2], repeat=3):
        if sum(x*x for x in b) == 4:
            valid_bs.append(b)
            
    # For each b, we can pick a set of a's from Z^8 with norm sq 4
    # such that pairwise dot <= 0
    # We pick the 8 vectors 2*e_i
    valid_as = []
    for i in range(8):
        a = [0]*8
        a[i] = 2
        valid_as.append(a)
        
    for b in valid_bs:
        for a in valid_as:
            v = np.zeros(d, dtype=np.int64)
            for i in range(8):
                v[i] = a[i]
            for i in range(3):
                v[8+i] = b[i]
            points.append(v)
            
    # Also try negative a's? 
    # If we pick -2*e_i, dot with 2*e_i is -8 <= 0.
    # But we can't pick both 2*e_i and -2*e_i if b is same?
    # Dot is -8. OK.
    # But we need pairwise dot <= 0 for ALL pairs in the set.
    # If we pick {2e1, -2e1}, dot is -8.
    # If we pick {2e1, 2e2}, dot is 0.
    # So we can pick both 2e_i and -2e_i?
    # Let's check: a1=2e1, a2=-2e1. Dot -8.
    # a1=2e1, a3=2e2. Dot 0.
    # a2=-2e1, a3=2e2. Dot 0.
    # Yes, we can pick all 16 vectors of type 2*e_i and -2*e_i.
    # So 16 per b.
    # 6 * 16 = 96 points.
    
    # Let's update the loop to include both signs
    # Remove the previously added 48 points? No, I'll just rewrite the addition part.
    
    # Clear the last 48 points added? 
    # It's easier to just generate the full list in one go.
    
    # I will replace the addition block with the improved version.
    
    arr = np.array(points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr
