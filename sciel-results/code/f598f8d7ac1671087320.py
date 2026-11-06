import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    N = 17
    MIN_DIST = 6
    
    # Helper: Convert int to bit vector (uint8)
    def int_to_vec(w):
        # Fast enough for 17 bits
        return np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
        
    # Helper: Convert list of ints to bit matrix (M x N)
    def ints_to_mat(words):
        return np.array([int_to_vec(w) for w in words], dtype=np.uint8)
        
    # Helper: Min distance from a new word to a matrix of words
    def min_dist(vec, mat):
        # vec: (N,), mat: (M, N)
        # Returns min hamming distance
        if len(mat) == 0:
            return N
        xor = mat ^ vec
        dists = xor.sum(axis=1)
        return int(np.min(dists))

    # Generate initial 32-word RM seed
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                if bit:
                    word |= (1 << point)
            # The 17th bit (index 16) is 0 for RM(1,4) extension? 
            # The prompt says "17-bit words". RM(1,4) is length 16.
            # The prompt says "seed has 32 words".
            # I will stick to the 16-bit part being the RM code, and bit 16 = 0.
            seed_words.append(word)
            
    # Verify seed validity
    seed_mat = ints_to_mat(seed_words)
    
    # Main Loop: Greedy + Local Search
    current_words = seed_words[:]
    current_mat = seed_mat
    
    # To speed up, keep the matrix updated.
    # We will use a list of ints for the "words" and a matrix for "bits".
    
    best_words = current_words[:]
    best_size = len(current_words)
    
    # Pre-compute bit masks for random generation
    all_masks = [1 << i for i in range(N)]
    
    attempts = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        # 1. Try to add a new word
        # Random generation
        new_word = random.getrandbits(N)
        
        # Check validity
        if min_dist(int_to_vec(new_word), current_mat) >= MIN_DIST:
            # Add it
            current_words.append(new_word)
            current_mat = np.vstack([current_mat, int_to_vec(new_word)])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
                
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        # 2. Local Search / Swap Strategy
        # If we are stuck for a while, try to swap an existing word with a new one
        if stuck_counter > 20:
            stuck_counter = 0
            # Pick a random index to remove
            idx = random.randint(0, len(current_words) - 1)
            old_word = current_words[idx]
            
            # Generate a candidate to replace it
            # Strategy: Flip 1-3 bits of a random word, or just pick a random word
            if random.random() < 0.5:
                # Mutate a random existing word
                base = random.choice(current_words)
                flips = random.randint(1, 3)
                new_word = base
                for _ in range(flips):
                    pos = random.randint(0, N - 1)
                    new_word ^= (1 << pos)
            else:
                new_word = random.getrandbits(N)
                
            # Check if new_word is valid against ALL other words (excluding idx)
            # We need to check distance to all j != idx
            # Temporarily remove idx from matrix for check? 
            # Or check against full matrix and ignore idx's distance?
            # min_dist returns min over all. If min < MIN_DIST, it might be due to idx.
            # We need: min(d(new, w_j) for j != idx) >= MIN_DIST
            
            # Efficient check:
            # Create a mask of indices? 
            # Simpler: Just compute distances to all, and check if the *minimum* among j!=idx is >= MIN_DIST.
            
            vec_new = int_to_vec(new_word)
            dists = (current_mat ^ vec_new).sum(axis=1)
            
            # Set distance to idx to infinity (or a large number)
            # We want to see if any OTHER word is too close.
            # dists[idx] is the distance to itself (if same word) or whatever.
            # We ignore idx.
            
            # Find min distance excluding idx
            if len(dists) > 1:
                # Create a view excluding idx? 
                # Just iterate? No, vectorize.
                # dists_no_idx = np.delete(dists, idx)
                # if np.min(dists_no_idx) >= MIN_DIST:
                #     valid = True
                # else:
                #     valid = False
                
                # Optimization: If the global min is >= MIN_DIST, it's definitely valid.
                # If the global min < MIN_DIST, check if the min index is idx.
                min_val = np.min(dists)
                if min_val >= MIN_DIST:
                    valid = True
                else:
                    min_idx = np.argmin(dists)
                    if min_idx == idx:
                        # The only conflict is with the word we are replacing.
                        # Check the second minimum?
                        # Or just check all others.
                        # For N=17, M~100, this is fast.
                        # But let's do it properly.
                        # Check if any j != idx has dist < MIN_DIST
                        valid = True
                        for j in range(len(dists)):
                            if j != idx and dists[j] < MIN_DIST:
                                valid = False
                                break
                    else:
                        valid = False
            else:
                valid = True
                
            if valid:
                # Perform Swap
                current_words[idx] = new_word
                current_mat[idx] = vec_new
                # Size doesn't change, but we might have improved "packing" 
                # allowing future additions.
                # We don't update best_size here as size is same.
                # But we reset stuck_counter? 
                # Actually, a successful swap might allow adding later.
                
        # 3. Aggressive Addition (if we haven't added for a long time)
        # The random generation above is very slow to find valid words as the code gets dense.
        # We should try to find a word that is "far" from existing ones.
        # Heuristic: 
        # Pick a random existing word w. 
        # Generate a word at distance ~ N/2 from w.
        # This is effectively what random does, but maybe biased.
        
        # Alternative: Use a "greedy" candidate generation
        # Pick 2-3 random existing words, compute their "average" or XORs, and perturb.
        
        if stuck_counter > 50:
            # Try a more structured candidate
            # Pick a random word w0
            w0 = random.choice(current_words)
            # Flip 6-10 bits (approx half)
            num_flips = random.randint(6, 10)
            new_word = w0
            # Randomly choose positions to flip
            positions = random.sample(range(N), num_flips)
            mask = 0
            for p in positions:
                mask |= (1 << p)
            new_word ^= mask
            
            # Check validity (same as above, but adding)
            if min_dist(int_to_vec(new_word), current_mat) >= MIN_DIST:
                current_words.append(new_word)
                current_mat = np.vstack([current_mat, int_to_vec(new_word)])
                if len(current_words) > best_size:
                    best_size = len(current_words)
                    best_words = current_words[:]
                stuck_counter = 0

    return best_words
