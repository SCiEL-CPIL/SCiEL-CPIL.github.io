import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0  # Safety margin before 60s limit

    # Start with the Reed-Muller construction (32 words)
    # RM(1,4) has length 16. We extend to 18 by adding 2 zero bits.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Pad to 18 bits (add 2 zeros at the end, i.e., high bits)
            words.append(word)
    
    # Convert to numpy array of bits for fast XOR and sum
    # Shape: (M, 18)
    M = len(words)
    codebook = np.zeros((M, N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            codebook[i, j] = (w >> j) & 1
            
    # List of integers for final return
    word_list = words[:]
    
    rng = np.random.default_rng(42)
    
    # Precompute a bit-to-index lookup for fast vectorization of candidates
    # We will process candidates in batches.
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        batch_size = 2000
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        # Convert batch to binary matrix (batch_size, N)
        # Using bit operations: (candidates[:, None] >> np.arange(N)) & 1
        # This is vectorized and fast.
        cand_matrix = (candidates[:, None] >> np.arange(N)) & 1
        cand_matrix = cand_matrix.astype(np.uint8)
        
        # Compute pairwise distances between each candidate and all existing words
        # xors: (batch_size, M, N) -> This might be too large for memory if M is big.
        # M starts at 32. Even if M grows to 100, batch_size=2000, 2000*100*18 ~ 3.6M bytes. Fine.
        # If M grows to 500, 2000*500*18 ~ 18M bytes. Fine.
        # If M grows to 1000, 2000*1000*18 ~ 36M bytes. Fine.
        
        # To avoid huge memory spikes, we can process in smaller chunks if M gets very large.
        # But let's assume M stays reasonable for now.
        
        # We need min distance for each candidate.
        # xors = codebook[None, :, :] ^ cand_matrix[:, None, :]
        # distances = np.sum(xors, axis=2)
        # min_dists = np.min(distances, axis=1)
        
        # This creates a (batch_size, M) array of distances.
        # Let's do it in a memory-efficient way if possible, or just rely on NumPy's speed.
        
        # Optimization: We can break early if we find a candidate that fails?
        # No, we want to find ALL valid candidates in the batch to add them all.
        # But adding them one by one changes the codebook.
        # If we add a candidate, it might invalidate other candidates in the same batch.
        # So we must check against the *current* codebook.
        # If we add multiple, we should probably add them one by one and update codebook.
        # Or, we can find all candidates that are valid against the *current* codebook,
        # then add them one by one, re-checking against the *updated* codebook?
        # That's complex.
        
        # Simpler: Iterate through candidates. For each, check against current codebook.
        # If valid, add to codebook.
        # This is O(batch_size * M * N).
        # With vectorization, the check is fast.
        
        # Let's try a hybrid:
        # 1. Compute min distance of each candidate against current codebook.
        # 2. Identify candidates with min_dist >= MIN_DIST.
        # 3. For those, add them one by one.
        #    When adding, we don't re-check against other new candidates in the batch,
        #    which might lead to invalid pairs among the new ones.
        #    To be safe, we should check new candidates against the *updated* codebook.
        
        # Let's stick to the safe loop but vectorize the inner check.
        
        # Actually, the loop in the current program is:
        # for cand in candidates:
        #    ... vectorized check against codebook ...
        #    if valid: add
        
        # This is already O(batch_size * M * N) but with NumPy speed for the M*N part.
        # The bottleneck is the Python loop over batch_size.
        
        # Can we vectorize the loop?
        # We can find all valid candidates in the batch against the *current* codebook.
        # Then we need to resolve conflicts among them.
        # If we add them all at once, we might violate distance between new words.
        
        # Strategy:
        # 1. Find all candidates in batch that are valid against current codebook.
        # 2. Among these, greedily select a subset that is pairwise valid.
        # 3. Add this subset to the codebook.
        
        # Step 1: Vectorized check
        # xors = codebook[None, :, :] ^ cand_matrix[:, None, :]
        # distances = np.sum(xors, axis=2)
        # min_dists = np.min(distances, axis=1)
        # valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        # This is fast.
        
        # Step 2: Greedy selection among valid_indices
        # We need to check pairwise distances among the candidates at valid_indices.
        # Let V be the number of valid candidates.
        # If V is small, we can do a simple loop.
        # If V is large, we might need to be careful.
        
        # Let's try this.
        
        if M > 0:
            xors = codebook[None, :, :] ^ cand_matrix[:, None, :]
            distances = np.sum(xors, axis=2)
            min_dists = np.min(distances, axis=1)
            valid_indices = np.where(min_dists >= MIN_DIST)[0]
        else:
            valid_indices = np.arange(batch_size)
        
        # Now, process valid_indices
        # We want to add as many as possible.
        # Greedy: pick one, remove it and all conflicting ones, repeat.
        
        # To make this efficient, we can check pairwise distances among the valid candidates.
        # But if V is large (e.g., 100), V^2 is 10000. Fine.
        
        if len(valid_indices) > 0:
            # Extract the valid candidate vectors
            valid_cands = cand_matrix[valid_indices]
            valid_ints = candidates[valid_indices]
            
            # Greedy selection
            selected = []
            # We'll use a list of booleans to track which are still available
            available = np.ones(len(valid_cands), dtype=bool)
            
            # To speed up, we can precompute the pairwise distances among valid_cands?
            # Or just do it on the fly.
            
            # On-the-fly:
            # For i in range(len(valid_cands)):
            #   if not available[i]: continue
            #   selected.append(i)
            #   available[i] = False
            #   # Check against all j > i
            #   # xors = valid_cands[i] ^ valid_cands[i+1:]
            #   # d = np.sum(xors, axis=1)
            #   # conflicts = d < MIN_DIST
            #   # available[i+1:][conflicts] = False
            
            # This is O(V^2 * N).
            
            for i in range(len(valid_cands)):
                if not available[i]:
                    continue
                
                # Select candidate i
                selected.append(i)
                available[i] = False
                
                # Check against remaining candidates
                if i + 1 < len(valid_cands):
                    # xors = valid_cands[i] ^ valid_cands[i+1:]
                    xors = valid_cands[i] ^ valid_cands[i+1:]
                    d = np.sum(xors, axis=1)
                    conflicts = d < MIN_DIST
                    available[i+1:][conflicts] = False
            
            # Add selected candidates to codebook
            if selected:
                new_rows = valid_cands[selected]
                new_ints = valid_ints[selected]
                
                # Append to codebook
                codebook = np.vstack([codebook, new_rows])
                word_list.extend(new_ints.tolist())
                M = len(word_list)
                
    return word_list
