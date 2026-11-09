import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0 # Safety margin

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    vectors = []
    
    # Part 1: All 220 vectors of Type 1 (Two +/- 2s)
    # These are always mutually compatible and compatible with Type 2.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 2: Greedy selection of Type 2 vectors (Eight +/- 1s)
    # We want a subset of these with pairwise dot product <= 4.
    # We start with the 128 from the seed (first 8 dims, even sum of +1s)
    # and try to add more.
    
    # Generate all supports for Type 2 (8 non-zero entries out of 11)
    # There are C(11, 8) = 165 supports.
    # For each support, there are 2^8 = 256 sign patterns.
    # Total 42240 vectors.
    
    # To keep it fast, we will iterate through a large random sample or structured set.
    # Let's try to include as many as possible.
    
    # Initialize with the seed's 128 vectors
    for signs in itertools.product([-1, 1], repeat=8):
        if signs.count(1) % 2 == 0:
            v = list(signs) + [0, 0, 0]
            vectors.append(v)
            
    # Now try to add more Type 2 vectors.
    # We will generate candidates and check against the current list.
    # To avoid O(N^2) on huge N, we limit the number of candidates we test.
    
    # Generate all Type 2 vectors? 42k is manageable if we check efficiently.
    # But checking against 252 + k vectors for each candidate is O(k).
    # If we add 1000 vectors, we do 42k * 1000 = 42M ops. Might be slow in Python.
    
    # Let's use a more efficient check or limit the pool.
    # We can group candidates by support.
    
    # For each support, we can pick a few sign patterns.
    # Let's try to pick one sign pattern per support first? 165 vectors.
    # Then try to add more.
    
    # Better: Iterate through all 42240 vectors, but break if time runs out.
    # We'll use a random order to find diverse vectors.
    
    all_type2 = []
    # Generate all Type 2 vectors
    for support in itertools.combinations(range(D), 8):
        # Create base vector with 1s at support
        base = [0] * D
        for idx in support:
            base[idx] = 1
            
        # Generate all sign patterns
        # To save time, we don't generate all 256 for every support if we are time-constrained.
        # But 165 * 256 = 42240. It's fine to generate them.
        for signs in itertools.product([-1, 1], repeat=8):
            v = base.copy()
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            all_type2.append(v)
            
    # Shuffle to avoid bias
    np.random.shuffle(all_type2)
    
    # Greedy add
    current_count = len(vectors)
    # Convert vectors to numpy array for faster dot products?
    # Or just use list of lists.
    
    # We will check candidates against the current `vectors` list.
    # To speed up, we can check against the last added few first?
    # No, must check against all.
    
    # Optimization: If we have many vectors, checking is slow.
    # We'll limit the number of candidates we attempt to add based on time.
    
    for v in all_type2:
        if time.time() > DEADLINE:
            break
            
        # Check if v is compatible with all existing vectors
        # We only need to check against other Type 2 vectors?
        # We proved Type 2 is compatible with Type 1.
        # So we only need to check against existing Type 2 vectors.
        # Let's separate them or just check all.
        # Checking all is safer.
        
        compatible = True
        for existing in vectors:
            # Dot product
            dot = 0
            # Unroll for speed?
            for i in range(D):
                dot += v[i] * existing[i]
            if dot > 4:
                compatible = False
                break
                
        if compatible:
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
