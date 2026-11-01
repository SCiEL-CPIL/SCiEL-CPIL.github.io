import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for the evaluator's own checks
    
    # The problem is specific to q=3, n=8, radius=1.
    # We can hardcode the initial structure for speed if parameters match, 
    # but a generic greedy is safer and fast enough with vectorization.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed-like behavior or simple greedy for other params
        # For this specific task, we assume the standard parameters.
        pass

    # 1. Initialize with the direct sum code (2187 words)
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all 2187 words
    # Shape: (2187, 8)
    # We use np.repeat and np.tile for efficiency
    tetracode_arr = np.array(tetracode) # (9, 4)
    
    # Generate all tails of length 4
    # There are 3^4 = 81 tails
    tails = np.array(np.meshgrid(*([np.arange(3)]*4))).T.reshape(-1, 4)
    
    # Construct full words
    # Repeat tetracode 81 times, tile tails 9 times
    code = np.hstack([
        np.repeat(tetracode_arr, 81, axis=0), 
        np.tile(tails, (9, 1))
    ])
    
    # 2. Compute coverage
    # Total words in space: 3^8 = 6561
    all_words = np.array(np.meshgrid(*([np.arange(3)]*8))).T.reshape(-1, 8)
    
    # Calculate Hamming distances from current code to all words
    # dist[i, j] = Hamming distance between code[i] and all_words[j]
    # We want to keep track of the minimum distance for each word j.
    # min_dist[j] = min_i dist(i, j)
    
    # Vectorized distance calculation
    # code: (2187, 8), all_words: (6561, 8)
    # We can compute distances in chunks to save memory, or just do it all if memory allows.
    # 2187 * 6561 * 8 bytes ~ 115 MB. This is fine.
    
    # However, doing full matrix multiplication or broadcasting might be slow in Python loop.
    # Let's use a chunked approach for distance calculation to be safe on memory and cache.
    
    min_dist = np.full(6561, 8, dtype=np.int8)
    
    # We will update min_dist as we add/remove codewords.
    # Actually, for the greedy set cover, we need to know which words are covered by which codeword.
    # A more efficient greedy for covering codes:
    # 1. Start with a set of codewords C.
    # 2. Identify "uncovered" words (dist > radius).
    # 3. Find codeword c that covers the most uncovered words.
    # 4. Add c to C? No, usually we start with a superset and remove, or start empty and add.
    #    Starting with the 2187 words and removing is better because we already have a valid cover.
    #    We want to remove a word if its removal doesn't leave any word uncovered.
    #    i.e., for every word w, if dist(w, c_removed) <= 1, there must be another c' in C \ {c_removed} with dist(w, c') <= 1.
    
    # This "removal" strategy is often faster to converge because we start with a valid solution.
    # Let's implement a greedy removal algorithm.
    
    # Data structures:
    # code: list of np arrays (current codewords)
    # We need to efficiently check if a word is covered by *other* codewords.
    
    # Let's try a different angle: Greedy ADD from a small seed?
    # The seed 2187 is too big.
    # Known good codes are ~486.
    # Maybe start with a random subset of 1000 words, then optimize?
    # Or use the structure of the tetracode?
    
    # Let's stick to the removal heuristic from the 2187 words.
    # It's deterministic and robust.
    
    # Precompute neighbors for each of the 6561 words?
    # A word w is covered by c if dist(w, c) <= 1.
    # This means c is w or c is w with one symbol changed.
    # So for each word w, the set of codewords that cover it is {w} U {neighbors of w}.
    
    # Let's represent the code as a set of indices in the 6561 space?
    # No, the code is a subset of the 6561 words.
    # Let's map each word to an index 0..6560.
    
    # Index of a word (w0..w7) is sum(w_i * 3^i).
    # We can create a lookup table: index -> word array.
    
    words_to_idx = {}
    for i in range(6561):
        # Convert base-3 representation
        w = []
        rem = i
        for _ in range(8):
            w.append(rem % 3)
            rem //= 3
        words_to_idx[tuple(w)] = i
        
    # This is slow. Let's just use the array `all_words` and find indices.
    # Actually, we can work directly with the 6561 array.
    
    # Current Code: set of indices into `all_words`
    # Initial code: the 2187 words generated above.
    # We need to map these 2187 words to their indices in `all_words`.
    
    # Since `all_words` contains all 6561 words, we can find the index of each code word.
    # But mapping 2187 words to indices might take time.
    
    # Alternative: Work with the arrays directly.
    # `code` is a list of arrays.
    # `covered_count[w]` = number of codewords within distance 1 of w.
    # If we remove a codeword c, we decrement covered_count[w] for all w within distance 1 of c.
    # We can remove c if min(covered_count) > 0 after the decrement.
    # i.e., if for all w, covered_count[w] >= 2 initially? No.
    # If covered_count[w] == 1 and the only coverer is c, we cannot remove c.
    
    # So:
    # 1. Compute `covered_count` for the initial 2187 words.
    # 2. Find a codeword c that can be removed (i.e., for all w it covers, covered_count[w] >= 2).
    # 3. Remove it, update counts.
    # 4. Repeat.
    
    # How to compute covered_count efficiently?
    # For each codeword c in code:
    #   For each neighbor w of c (dist <= 1):
    #      covered_count[w] += 1
    
    # Number of neighbors for a word in q=3, n=8, r=1:
    # 1 (self) + 8 * 2 (change 1 pos to 1 of 2 other vals) = 17.
    # Total updates: 2187 * 17 ~ 37000. Very fast.
    
    # We need to map `all_words` indices to counts.
    # Let's create an array `counts` of size 6561.
    
    # First, we need to know the index of each word in `all_words`.
    # We can create a dictionary mapping tuple(word) -> index.
    # Or, since we generated `all_words` in a specific order (meshgrid), we can reconstruct the index.
    # meshgrid order: 
    # w = [w0, w1, ..., w7]
    # index = w0 * 3^7 + w1 * 3^6 + ... + w7 * 3^0 ?
    # Let's check np.meshgrid default indexing.
    # np.meshgrid(*([np.arange(3)]*8)).T.reshape(-1, 8)
    # The last dimension varies fastest.
    # So index = w0 * 3^7 + w1 * 3^6 + ... + w7.
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    # Precompute neighbors for each word index?
    # We only need neighbors for words that are in the code.
    # But we need to update counts for neighbors.
    
    # Let's precompute the list of neighbor indices for every possible word?
    # 6561 * 17 ~ 110k integers. Small.
    # neighbors[i] = list of indices j such that dist(all_words[i], all_words[j]) <= 1
    
    neighbors = [[] for _ in range(6561)]
    for i in range(6561):
        w = all_words[i]
        # Self
        neighbors[i].append(i)
        # Neighbors
        for pos in range(8):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Create new word
                # Modifying a copy is slow. Let's calculate index directly?
                # Or just construct the array and lookup?
                # Lookup in dict is O(1).
                # Let's build the dict first.
                
    # Build word_to_idx dict
    word_to_idx = {}
    for i in range(6561):
        word_to_idx[tuple(all_words[i])] = i
        
    # Precompute neighbors
    for i in range(6561):
        w = all_words[i]
        neighbors[i].append(i)
        for pos in range(8):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Construct new word tuple
                # w is a numpy array
                # w_list = list(w)
                # w_list[pos] = new_val
                # idx = word_to_idx[tuple(w_list)]
                # neighbors[i].append(idx)
                
                # Optimization: calculate index directly
                # idx = i + (new_val - orig_val) * (3 ** (7 - pos))
                # This works because base-3 representation is linear in this way.
                # Let's verify.
                # i = sum(w_j * 3^(7-j))
                # i' = sum(w'_j * 3^(7-j))
                # w'_pos = new_val, w'_j = w_j for j != pos
                # i' - i = (new_val - orig_val) * 3^(7-pos)
                # Yes.
                idx = i + (new_val - orig_val) * (3 ** (7 - pos))
                neighbors[i].append(idx)
                
    # 1. Initialize counts
    counts = np.zeros(6561, dtype=np.int32)
    
    # 2. Add initial code words
    # code_indices: list of indices in all_words
    code_indices = []
    for c in code:
        idx = word_to_idx[tuple(c)]
        code_indices.append(idx)
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    # 3. Greedy Removal
    # We maintain a list of current code indices.
    # We want to remove an index `idx` if for all nb in neighbors[idx], counts[nb] > 1.
    # Note: If we remove idx, we must decrement counts[nb] for all nb in neighbors[idx].
    
    # To find a removable index efficiently:
    # We can check candidates.
    # A good heuristic: remove the one with the "least" unique coverage?
    # Or just iterate through the list and check.
    # Since we start with 2187 and end with ~486, we do ~1700 removals.
    # Checking all remaining candidates each time might be O(N^2).
    # 2000^2 = 4M checks. Each check is 17 ops. 68M ops.
    # Python loop might be slow. 120s is plenty for 68M simple ops in C, but Python is slower.
    # We should vectorize or optimize the search for removable candidates.
    
    # Alternative:
    # Maintain a list of "critical" words (counts == 1).
    # If a word w has count 1, the codeword covering it is "essential".
    # We cannot remove any codeword that covers an essential word.
    # So, we can only remove codewords that do NOT cover any essential word.
    
    # Let's track `essential_words` = set of indices w where counts[w] == 1.
    # And for each codeword c, track if it covers any essential word.
    
    # Actually, simpler:
    # Iterate through code_indices.
    # For each c, check if any nb in neighbors[c] has counts[nb] == 1.
    # If not, remove it.
    
    # To speed this up, we can precompute for each c a flag `is_removable`.
    # But counts change.
    
    # Let's try a simple loop first and see if it's fast enough.
    # 1700 removals * 2000 candidates * 17 neighbors = 57M iterations.
    # In Python, this might take 5-10 seconds. Acceptable.
    
    # We need to be careful with the time limit.
    
    while time.time() - start_time < timeout:
        removed_any = False
        # We iterate over a copy of the list to allow removal
        # Actually, we can just build a new list.
        
        new_code_indices = []
        for idx in code_indices:
            # Check if idx can be removed
            removable = True
            for nb in neighbors[idx]:
                if counts[nb] == 1:
                    removable = False
                    break
            if removable:
                # Remove idx
                for nb in neighbors[idx]:
                    counts[nb] -= 1
                removed_any = True
            else:
                new_code_indices.append(idx)
                
        if not removed_any:
            break
        
        code_indices = new_code_indices
        
    # Convert back to list of lists
    result = [all_words[i].tolist() for i in code_indices]
    
    return result

