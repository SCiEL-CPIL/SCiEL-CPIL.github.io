import numpy as np
from itertools import combinations

def construct():
    # The seed program constructs a [17, 8, 6] code (256 words) by shortening the [18, 9, 6] extended QR code.
    # To beat the seed score of 4.0 (value 256 / seed 64? No, seed score 4.0 implies value/seed=4.0.
    # If seed program returns 256 words, and score is 4.0, then "seed" in the score formula might be 64?
    # Or maybe the seed program in the prompt actually returns 256, and the "seed" value for comparison is 64?
    # Wait, the prompt says: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate; the seed has 32 words."
    # Ah, the seed program *has* 32 words? No, the text says "the seed has 32 words" in the description of the task, but the CURRENT PROGRAM code generates 256 words (it iterates 256 messages, filters parity, likely keeps ~128 or 256? Let's check the code).
    # The current code iterates `msg in range(256)`. It computes `word` of 17 bits. It computes `parity`. It appends `word` if `parity == 0`.
    # If the underlying code is [17, 9, 5], there are 512 words. Half are even weight -> 256 words.
    # So the current program returns 256 words.
    # The task description says: "Score: number of codewords, maximized." and "search score is value/seed, so 1.0 equals the seed program".
    # If current score is 4.0000, and it returns 256 words, then `seed` must be 64.
    # Why is the seed 64? Maybe the reference seed program is a simpler one?
    # Regardless, my goal is to maximize the number of codewords (value).
    # Current value: 256.
    # Target: A_2(17,6) is between 258 and 340.
    # I need to find a code with > 256 words.
    
    # Strategy:
    # 1. Start with the 256-word [17, 8, 6] code (the shortened extended QR code).
    # 2. Try to add more codewords to this set.
    # 3. The set of all valid codewords is the set of 17-bit integers with pairwise distance >= 6.
    # 4. We can use a greedy search or local search to add words.
    # 5. A powerful approach: Use the 256-word code as a base. Compute the "distance profile" or simply check all possible 17-bit words (2^17 = 131072) to see if they can be added.
    #    Actually, checking all 131072 words against 256 existing words is 131072 * 256 * 17 ops ~ 500 million ops. In Python this might be slow (seconds).
    #    We can optimize by precomputing distances or using bit operations.
    #    Better: Use a greedy algorithm. Start with the 256 words. Then iterate through all other 17-bit words.
    #    For each word, check if it is at distance >= 6 from all current codewords.
    #    If yes, add it.
    #    This is O(N * 2^N * N) in worst case, but N=256, 2^N=131072.
    #    131072 * 256 * 17 ~ 570 million bit operations. Python is slow.
    #    We can vectorize with NumPy.
    
    # Step 1: Generate the 256-word base code.
    # We can reuse the logic from the current program or a simpler construction.
    # The [17, 8, 6] code is the even-weight subcode of the [17, 9, 5] QR code.
    # Let's just use the code provided in the prompt as the base, but ensure it's correct.
    # Actually, I can just hardcode the generator or use the GF(2^8) method.
    # Let's use the GF(2^8) method from the prompt to generate the 256 words.
    
    poly = 0x11D
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return p

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            n >>= 1
        return result

    alpha = 3
    beta = gf_pow(alpha, 15)
    
    p = [1]
    for i in range(8):
        r = gf_pow(beta, (1 << i))
        new_p = [0] * (len(p) + 1)
        for j in range(len(p)):
            new_p[j] ^= gf_mul(p[j], r)
            new_p[j+1] ^= p[j]
        p = new_p
        
    # p is coeffs of g(x) = c0 + c1 x + ... + c8 x^8
    # Generate [17, 9, 5] code
    base_codewords = []
    for msg in range(512): # k=9, so 512 messages
        m = [(msg >> i) & 1 for i in range(9)]
        h = [0] * 17
        for i in range(9):
            if m[i] == 1:
                for j in range(9):
                    if p[j] == 1:
                        h[i+j] ^= 1
        word = 0
        for i in range(17):
            if h[i]:
                word |= (1 << i)
        # Shorten by fixing parity bit to 0?
        # The prompt says: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate"
        # The [18,9,6] code is the extension of [17,9,5].
        # Shortening [18,9,6] at the parity coordinate gives the even-weight words of [17,9,5].
        # So we keep only words with even weight.
        if word.bit_count() % 2 == 0:
            base_codewords.append(word)
            
    # Now we have 256 codewords.
    # We want to add more.
    # Let's create a set of current codewords for fast lookup.
    current_set = set(base_codewords)
    
    # To efficiently check distances, we can precompute the bit vectors.
    # But 2^17 is small enough that we can just iterate.
    # Optimization: 
    # For a candidate word `w`, we need to check `dist(w, c) >= 6` for all `c` in `current_set`.
    # `dist(w, c) = (w ^ c).bit_count()`
    # If we find any `c` with `dist < 6`, we reject `w`.
    
    # We can speed this up by checking against a subset first, or using NumPy.
    # Let's use NumPy for vectorized distance checking.
    
    codewords_array = np.array(list(current_set), dtype=np.uint8)
    # Shape: (256, 17)
    # We need to represent each codeword as 17 bits.
    # Let's create a matrix of bits.
    bits = np.zeros((len(codewords_array), 17), dtype=np.uint8)
    for i, w in enumerate(codewords_array):
        for b in range(17):
            if (w >> b) & 1:
                bits[i, b] = 1
                
    # Iterate through all possible 17-bit words
    # There are 131072 words.
    # For each word, compute distance to all 256 codewords.
    # Min distance must be >= 6.
    
    # We can vectorize the check for all candidates at once?
    # 131072 * 256 * 17 is too big for a single matrix (131072 * 256 = 33 million rows, 17 cols -> 500 million bytes).
    # It's manageable but might be slow in Python if done in a loop.
    # Let's do it in batches.
    
    new_codewords = []
    
    # Pre-compute bit masks for the 17 positions
    masks = [1 << i for i in range(17)]
    
    # To speed up, we can sort candidates by weight?
    # No, let's just iterate.
    
    # Optimization: Use a "forbidden" set?
    # If a word `w` is rejected because it's too close to `c`, then any word `w'` that is also close to `c` might be rejected.
    # But the condition is pairwise.
    
    # Let's try a greedy approach with NumPy.
    # We will process candidates in chunks.
    
    all_candidates = np.arange(2**17, dtype=np.uint32)
    
    # Filter out existing codewords
    existing_mask = np.isin(all_candidates, list(current_set))
    candidates = all_candidates[~existing_mask]
    
    # We want to add as many as possible.
    # Greedy: Pick the candidate that is "most compatible" or just the first one that fits.
    # Since we want to maximize count, any valid addition is good.
    # Let's just find all valid additions to the *initial* set.
    # Note: Adding one word might invalidate another candidate.
    # So we must add them one by one and update the set.
    
    # To make this fast, we can maintain the array of current codewords.
    # Start with the 256 words.
    current_codewords = list(base_codewords)
    
    # For efficiency, we can pre-compute the distances of all candidates to the initial 256 words.
    # dist_matrix[i, j] = distance(candidate[i], codeword[j])
    # This is 131072 * 256 = 33 million entries.
    # Each entry is a byte (max dist 17).
    # 33 MB. This is fine.
    
    # Let's compute the min distance of each candidate to the initial set.
    # If min_dist >= 6, it's a candidate for addition.
    
    # We can compute this in chunks to save memory.
    
    valid_candidates = []
    
    chunk_size = 10000
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        chunk = candidates[start:end]
        
        # Convert chunk to bit matrix
        # chunk is uint32 array
        # Create bit matrix (chunk_size, 17)
        chunk_bits = np.zeros((len(chunk), 17), dtype=np.uint8)
        for b in range(17):
            chunk_bits[:, b] = (chunk >> b) & 1
            
        # Compute distances to current_codewords (bits matrix)
        # dist = sum(chunk_bits[:, b] != bits[:, b] for b in 17)
        # Using broadcasting:
        # chunk_bits is (C, 17), bits is (N, 17)
        # We want (C, N) matrix of distances.
        # diff = chunk_bits[:, :, None] ^ bits[None, :, :]  # (C, N, 17)
        # dists = np.sum(diff, axis=2)
        
        # This creates a (C, N, 17) array.
        # C=10000, N=256, 17.
        # 10000 * 256 * 17 = 43 million bytes. ~43 MB.
        # This is acceptable.
        
        # However, Python overhead for creating this might be high.
        # Alternative: Use the fact that dist(u, v) = wt(u) + wt(v) - 2 * dot(u, v)
        # wt(chunk) is precomputable.
        # wt(bits) is precomputable.
        # dot(chunk_bits, bits.T) gives (C, N) matrix of common 1s.
        
        chunk_wt = np.sum(chunk_bits, axis=1)
        code_wt = np.sum(bits, axis=1)
        
        # dot product
        # chunk_bits is uint8, bits is uint8
        # dot product of 0/1 vectors is count of common 1s.
        common_ones = chunk_bits @ bits.T
        
        dists = chunk_wt[:, None] + code_wt[None, :] - 2 * common_ones
        
        # Check if min distance >= 6
        min_dists = np.min(dists, axis=1)
        
        # Indices in chunk where min_dist >= 6
        valid_indices = np.where(min_dists >= 6)[0]
        
        for idx in valid_indices:
            valid_candidates.append(chunk[idx])
            
    # Now we have a list of candidates that are valid against the *initial* 256 words.
    # But they might not be valid against each other, or against the words we add.
    # We need to greedily add them.
    
    # Sort valid_candidates? Maybe by weight or index?
    # Let's just iterate through them.
    
    # To speed up the greedy addition, we can maintain the bit matrix of current codewords.
    # But updating it dynamically is annoying.
    # Given the small number of additions expected (maybe a few dozen),
    # we can just check against the full current set for each addition.
    
    # Current set size: 256.
    # Number of valid candidates: likely a few thousand.
    # For each candidate, check against 256 + k words.
    # If we add 100 words, we check against 356 words.
    # 3000 candidates * 356 checks * 17 bits ~ 18 million ops. Fast.
    
    final_codewords = list(base_codewords)
    final_set = set(final_codewords)
    
    # We can optimize the check by keeping the bit matrix of final_codewords.
    # But rebuilding it is O(N).
    # Let's just use the set and check distances.
    
    for cand in valid_candidates:
        # Check if cand is at distance >= 6 from all words in final_codewords
        valid = True
        for w in final_codewords:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            final_codewords.append(cand)
            final_set.add(cand)
            
    return final_codewords
