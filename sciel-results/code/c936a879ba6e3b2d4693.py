import numpy as np
import itertools
import time

D = 11

def _generate_vectors(M):
    """Generate all integer vectors in R^D with norm squared M."""
    vectors = []
    
    # Helper to generate combinations of indices and values
    # We iterate over possible weights and value distributions
    
    # For small M, we can just iterate over all possible non-zero patterns?
    # No, D=11 is too big for 2^11 * 3^11.
    # We generate based on weight.
    
    # Determine possible weights w and values v such that sum(v^2) = M
    # Max weight is min(D, M) (if values are 1)
    
    for w in range(1, min(D, M) + 1):
        # Find all multisets of w integers (squared) that sum to M
        # Values must be integers, so v^2 is a square number.
        # Possible squared values: 1, 4, 9, ... <= M
        
        # This is a partition problem.
        # For M=4, w=1: [4] -> values [2]
        # w=4: [1,1,1,1] -> values [1,1,1,1]
        
        # Let's just hardcode the generation for specific M or use a recursive generator for value squares.
        
        def gen_squares(sum_target, max_val, count):
            if count == 0:
                if sum_target == 0:
                    yield []
                return
            if sum_target < 0:
                return
            # Try all possible squares
            v = 1
            while v*v <= sum_target:
                # Use v*v
                for rest in gen_squares(sum_target - v*v, v, count-1):
                    yield [v*v] + rest
                v += 1
        
        # Get all possible lists of squared values of length w that sum to M
        # We need unique combinations of values (order doesn't matter for the set of values, but signs do)
        # Actually, we need to assign these values to indices.
        
        # Let's collect unique value patterns
        patterns = set()
        for sq_list in gen_squares(M, int(np.sqrt(M)), w):
            # Sort to handle permutations
            patterns.add(tuple(sorted(sq_list)))
        
        for pat in patterns:
            # For each pattern, generate all permutations of indices and signs
            # pat contains squared values. e.g., (1, 1, 4)
            # Values are sqrt(pat).
            # We need to assign these values to w distinct indices from D.
            
            # Get unique values and their counts?
            # No, just treat them as a list of magnitudes.
            mags = [int(np.sqrt(x)) for x in pat]
            
            # Choose w indices from D
            for indices in itertools.combinations(range(D), w):
                # For each index, assign a magnitude from mags
                # And a sign (+/-)
                # This generates duplicates if mags has repeated values.
                # To avoid duplicates, we can just generate all sign combinations
                # and all permutations of mags assigned to indices?
                # It's easier to just generate all vectors and use a set.
                
                # Iterate over all permutations of mags assigned to indices
                # And all sign combinations
                
                # To be efficient, we can just iterate over all 2^w signs
                # and all permutations of mags?
                # If mags has duplicates, permutations are redundant.
                
                # Let's just generate all possible vectors for these indices and mags
                # and add to a set to deduplicate.
                
                # Actually, simpler:
                # For a fixed set of indices, and a fixed multiset of magnitudes:
                # We can assign the magnitudes to the indices in all distinct ways.
                # Then assign signs.
                
                # Let's use a set of tuples to store vectors
                # But we are in a generator context? No, we are building a list.
                
                # Let's just append to a local list and deduplicate at the end of _generate_vectors
                # Or use a set of tuples.
                
                # For each permutation of mags:
                for perm in set(itertools.permutations(mags)):
                    # For each sign combination:
                    for signs in itertools.product([1, -1], repeat=w):
                        v = [0] * D
                        for i, idx in enumerate(indices):
                            v[idx] = perm[i] * signs[i]
                        vectors.append(tuple(v))
                        
    return list(set(vectors))

def _greedy_construct(vectors, M):
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Shuffle to avoid bias
    np.random.shuffle(vectors)
    
    selected = []
    selected_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    limit_dot = M / 2.0
    
    for v in vectors:
        v_arr = np.array(v, dtype=np.int64)
        
        if not selected:
            selected.append(v)
            selected_arr = v_arr.reshape(1, -1)
            continue
            
        # Compute dot products
        dots = selected_arr @ v_arr
        
        # Check if all dot products <= M/2
        # Using integer arithmetic: 2 * dot <= M
        if np.all(dots * 2 <= M):
            selected.append(v)
            selected_arr = np.vstack((selected_arr, v_arr.reshape(1, -1)))
            
    return np.array(selected, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_vectors = None
    best_k = 0
    
    # Seed solution
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(tuple(v))
    seed_arr = np.array(seed_vectors, dtype=np.int64)
    best_vectors = seed_arr
    best_k = len(seed_arr)
    
    # Try M=4
    if time.time() < deadline:
        vecs = _generate_vectors(4)
        res = _greedy_construct(vecs, 4)
        if len(res) > best_k:
            best_k = len(res)
            best_vectors = res
            
    # Try M=5
    if time.time() < deadline:
        vecs = _generate_vectors(5)
        res = _greedy_construct(vecs, 5)
        if len(res) > best_k:
            best_k = len(res)
            best_vectors = res
            
    # Try M=2 (just in case, though we know it's 220)
    if time.time() < deadline:
        vecs = _generate_vectors(2)
        res = _greedy_construct(vecs, 2)
        if len(res) > best_k:
            best_k = len(res)
            best_vectors = res

    return best_vectors
