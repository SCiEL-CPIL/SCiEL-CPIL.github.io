import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    
    # Type 1: Integer E8 minimal vectors scaled by 2
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    
    # Type 2: Half-integer E8 minimal vectors scaled by 2
    for mask in range(256):
        signs = []
        count_plus = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs.append(1)
                count_plus += 1
            else:
                signs.append(-1)
        
        if count_plus % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            vectors.append(v)
            
    # Type 3: D3 minimal vectors scaled by 2, padded to 11D
    # D3 minimal vectors are permutations of (+/-1, +/-1, 0)
    # Scaled by 2: (+/-2, +/-2, 0)
    d3_vectors = []
    # Generate all permutations of (2, 2, 0), (2, -2, 0), etc.
    # Easier: iterate over all 3! permutations of indices and signs
    # Or just hardcode the 12 vectors?
    # Let's generate them properly.
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        # perm gives the positions of the values in the tuple (v0, v1, v2)
        # We want to assign values to indices 0,1,2.
        # Let's iterate over the 4 sign patterns for the two non-zeros.
        # Pattern 1: (2, 2, 0) -> non-zeros are same sign.
        # Pattern 2: (2, -2, 0) -> non-zeros are opposite sign.
        # We need to place the 0 in one of the 3 positions.
        pass
    
    # Simpler generation for D3 scaled by 2:
    # Two non-zero entries are 2 or -2. One is 0.
    for i in range(3):
        for j in range(3):
            if i == j: continue
            k = 3 - i - j
            # i, j are indices of non-zeros. k is index of 0.
            # Signs for i and j
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    # This generates duplicates?
                    # i=0, j=1, s1=1, s2=1 -> (2,2,0)
                    # i=1, j=0, s1=1, s2=1 -> (2,2,0) at (1,0)? No, v[1]=2, v[0]=2. Same vector.
                    # So we will generate each vector twice (once for i,j and once for j,i).
                    # And for signs, (1,1) and (1,1) swapped is same.
                    # Actually, (i,j) and (j,i) with same signs produce the same vector.
                    # So we can just divide by 2? Or use a set.
                    # Or restrict i < j.
                    pass
    
    # Let's use i < j to avoid duplicates.
    d3_vecs = []
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    d3_vecs.append(v)
    
    # Now pad and add to main list
    for v in d3_vecs:
        v_11 = np.zeros(11, dtype=np.int64)
        v_11[8:11] = v
        vectors.append(v_11)
        
    # Pad E8 vectors to 11D
    points = np.zeros((len(vectors), 11), dtype=np.int64)
    for idx, v in enumerate(vectors):
        if len(v) == 8:
            points[idx, :8] = v
        else:
            points[idx, :] = v
            
    return points
