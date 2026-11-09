def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    def get_linear_code():
        # Greedy linear construction to find a [17,8,6] code (256 words)
        # We want a basis of size 8.
        basis = []
        code = [0]
        
        # Order vectors by weight to potentially find a "dense" code?
        # Or just iterate.
        # To be deterministic and fast, we iterate through all words.
        # However, checking 131k words against growing code is O(N * 2^N * 2^k).
        # 131k * 256 * 17 ops ~ 500M ops. Might be slow in Python.
        
        # Better: Use a known generator matrix for the [17,8,6] code.
        # The [17,8,6] code is the shortening of the [18,9,6] QR code.
        # Generator matrix G for [18,9,6] QR code (systematic part):
        # The quadratic residues mod 17 are {1, 2, 4, 8, 9, 13, 15, 16}.
        # The generator matrix can be constructed, but it's easier to use a standard one if available.
        # Since I don't have a hardcoded table, I will use a fast greedy search restricted to a subset.
        
        # Actually, a simple greedy search for a basis of size 8 is fast if we prune.
        # We only need to find 8 vectors.
        # We can iterate through words, and for each, check if it's compatible with current code.
        # If yes, add it and double the code.
        # We stop when we have 8 basis vectors.
        
        # To speed up, we can prioritize high weight words?
        # Or just iterate 1 to SIZE-1.
        
        candidates = list(range(1, SIZE))
        # Shuffle to allow random restarts if needed, but first try deterministic
        # Actually, the first 8 compatible words in a specific order might not form a full-rank basis?
        # No, if we add a vector, it must be linearly independent to increase the code size by factor 2.
        # The greedy algorithm that adds any valid vector automatically maintains linearity and independence if we manage the basis correctly.
        
        # Let's use a standard QR code generator matrix if I can recall/derive it.
        # Alternatively, just run the greedy. 256 words is small.
        # The bottleneck is the distance check.
        # We can use numpy for batch distance checks?
        # No, the code grows dynamically.
        
        # Let's try the greedy with a time limit.
        current_code = [0]
        basis = []
        
        # We will try to add vectors in order of their index.
        # To ensure we find the [17,8,6] code, we might need a specific order.
        # But any [17,8,6] code is equivalent.
        
        for w in candidates:
            if len(basis) == 8:
                break
            if time.time() > deadline - 50:
                break
            
            # Check if w is at distance >= 6 from all c in current_code
            # current_code size is 2^len(basis)
            valid = True
            # Optimization: check against basis vectors? No, must check against all codewords.
            # But d(w, c1 ^ c2) is not simply related to d(w, c1) and d(w, c2).
            # So we must check all.
            
            # Use a loop. Size is small (max 128 before adding 8th).
            for c in current_code:
                if weights[w ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(w)
                # Extend code
                # new_code = [c ^ w for c in current_code]
                # current_code.extend(new_code)
                # To save time, we can just keep the list.
                new_words = [c ^ w for c in current_code]
                current_code.extend(new_words)
                
        return current_code

    # Step 1: Get the 256-word linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        # Fallback: return what we have
        return lin_code

    # Step 2: Find candidates outside the linear code
    # A word w is a candidate if min_{c in lin_code} d(w, c) >= 6
    
    # To speed this up, we can convert lin_code to a numpy array
    lin_code_np = np.array(lin_code, dtype=np.int32)
    
    candidates = []
    
    # Iterate through all words
    # We can skip words in lin_code
    lin_code_set = set(lin_code)
    
    for w in range(SIZE):
        if w in lin_code_set:
            continue
        
        # Check distance to all lin_code words
        # dists = weights[w ^ lin_code_np]
        # if dists.min() >= MIN_D:
        
        # Manual loop might be faster for small arrays in Python if numpy overhead is high?
        # 256 elements.
        # Let's use numpy.
        dists = weights[w ^ lin_code_np]
        if dists.min() >= MIN_D:
            candidates.append(w)
            
    if not candidates:
        return lin_code

    # Step 3: Find maximum subset of candidates with pairwise distance >= 6
    # This is a Max Independent Set problem on a graph of candidates.
    # Given the small expected size of the solution (1 or 2), we can use greedy.
    
    best_subset = []
    
    # Try multiple random orders
    # We have plenty of time left (approx 30-40s).
    # Candidates list might be large.
    
    # Optimization: If we just want to beat 256, finding 1 candidate is enough for 257.
    # Finding 2 compatible candidates gives 258.
    # Let's try to find a large subset.
    
    # Precompute distances between candidates?
    # If |candidates| is large (e.g., 1000), pairwise is 1M checks.
    # weights[u ^ v] is fast.
    
    # Greedy with restarts
    max_found = 0
    best_set = []
    
    num_restarts = 100 # Can be increased if time permits
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        cands = candidates[:]
        random.shuffle(cands)
        
        current_set = []
        for w in cands:
            valid = True
            for u in current_set:
                if weights[w ^ u] < MIN_D:
                    valid = False
                    break
            if valid:
                current_set.append(w)
                
        if len(current_set) > max_found:
            max_found = len(current_set)
            best_set = current_set[:]
            
            # Early stop if we found a "good" number?
            # Milshtein says 258, so max_found should be at least 2.
            # If we find 2, we can stop? Or try to find more?
            # Let's keep going to see if we can find 3, 4...
            
    final_code = lin_code + best_set
    return final_code
