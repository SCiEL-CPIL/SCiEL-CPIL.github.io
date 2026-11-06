import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Pre-compute all vectors as uint8 for fast XOR-based distance
    all_indices = np.arange(TOTAL, dtype=np.uint32)
    # all_vectors[i, j] = bit j of i
    all_vectors = (all_indices[:, None] >> np.arange(N)[None, :]) & 1
    all_vectors = all_vectors.astype(np.uint8)
    
    # Seed: First-order Reed-Muller code RM(1,4) = 32 words of length 16,
    # embedded as 17-bit words (leading zero).
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
    
    seed_indices = np.array(seed_words, dtype=np.uint32)
    
    best_size = 0
    best_code = []
    
    # Use a "least conflicts" greedy heuristic with multiple restarts.
    # Instead of random order, we score candidates by how many conflicts they have
    # with the current code, and pick the one with minimum conflicts (that has zero conflicts).
    # But that's expensive. Instead, we use a hybrid: 
    #   - Maintain a set of "available" candidates (those not yet excluded).
    #   - At each step, sample a batch of candidates, compute their min-distance to current code,
    #     and add those that qualify. This is similar to random greedy but with batched checks.
    
    # Actually, the current approach (random greedy) already scores 4.3125.
    # Let's try a different structural approach: 
    #   1. Start with seed.
    #   2. Repeatedly find the word with the largest "room" (minimum distance to all codewords) 
    #      among a random sample, and add it if it qualifies.
    #   3. This is a "best-first" greedy within a random sample.
    
    # Alternative structural change: Use a conflict matrix approach.
    # For each pair (i, j), compute if dist(i, j) < 6. This is a 131072 x 131072 matrix - too big.
    # Instead, for each codeword in our code, maintain a bitset of words that conflict with it.
    # A word can be added if it doesn't conflict with any codeword.
    
    # Let's use a different representation: 
    #   - For each word w, precompute the set of words at distance < 6 from w.
    #   - This is too large to store explicitly.
    #   
    # Better: Use the fact that we're working with 17-bit words. 
    #   - For a given word w, the words at distance < 6 are those with Hamming weight of (w xor w') < 6.
    #   - We can enumerate all words at distance 0, 1, 2, 3, 4, 5 from w.
    #   - Total: C(17,0)+C(17,1)+...+C(17,5) = 1+17+136+680+2380+6188 = 9402 words.
    #   - For each codeword in our code, we can mark these 9402 words as "conflicting".
    #   - A new candidate is valid if it's not marked.
    
    # This is a structural change: instead of checking distance to all codewords,
    # we maintain a conflict set and check membership.
    
    # Precompute for each word, the list of words at distance <= 5.
    # This is 131072 * 9402 ~ 1.2 billion entries - too much memory.
    # Instead, we compute on the fly for each codeword we add.
    
    # Let's try a simpler structural change: 
    #   - Use a priority queue / best-first approach where we score candidates by 
    #     their minimum distance to the current code, and always pick the best.
    #   - This requires maintaining a score for each candidate, which is expensive.
    
    # Let's go with a different approach: 
    #   - Random greedy but with a smarter initialization and more aggressive restarts.
    #   - Key insight: The current random greedy works well because the code density is low.
    #   - Let's add a "local improvement" step: after a greedy pass, try to swap out 
    #     poorly-placed codewords for better ones.
    
    # Actually, let's try the conflict-set approach but more efficiently:
    #   - Maintain a boolean array `conflict` of size TOTAL.
    #   - When we add a codeword w, mark all words at distance < 6 from w as conflicting.
    #   - To find the next codeword, scan for the first unconflicting word.
    #   - This is O(TOTAL) per addition, but we can optimize by maintaining a list of available words.
    
    # Precompute neighbors: for each word, store the indices of words at distance 1..5.
    # This is feasible: 131072 words * 9401 neighbors ~ 1.2 billion - still too much.
    
    # Let's use a hybrid: 
    #   - Random greedy as before, but with a twist: 
    #     Instead of adding a word if it's valid, we compute its "score" = min distance to current code.
    #     We only add it if score >= 6. But we also track the top-K candidates by score.
    #     After the first pass, we do a second pass with the top-K candidates in a different order.
    
    # Let's just improve the random greedy with better batching and more restarts.
    # The key structural change: use a "conflict bitset" approach where we maintain 
    # a numpy boolean array of available candidates, and update it incrementally.
    
    best_size = 0
    best_code = []
    
    # Precompute: for each word, the list of words at distance exactly 1, 2, 3, 4, 5.
    # We'll compute this on the fly using bit operations.
    
    def get_neighbors(w, max_dist=5):
        """Return all words at Hamming distance 1..max_dist from w."""
        neighbors = []
        # Distance 1: flip one bit
        for bit in range(N):
            neighbors.append(w ^ (1 << bit))
        # Distance 2: flip two bits
        for bit1 in range(N):
            for bit2 in range(bit1 + 1, N):
                neighbors.append(w ^ (1 << bit1) ^ (1 << bit2))
        # Distance 3: flip three bits
        for bit1 in range(N):
            for bit2 in range(bit1 + 1, N):
                for bit3 in range(bit2 + 1, N):
                    neighbors.append(w ^ (1 << bit1) ^ (1 << bit2) ^ (1 << bit3))
        # Distance 4: flip four bits
        for bit1 in range(N):
            for bit2 in range(bit1 + 1, N):
                for bit3 in range(bit2 + 1, N):
                    for bit4 in range(bit3 + 1, N):
                        neighbors.append(w ^ (1 << bit1) ^ (1 << bit2) ^ (1 << bit3) ^ (1 << bit4))
        # Distance 5: flip five bits
        for bit1 in range(N):
            for bit2 in range(bit1 + 1, N):
                for bit3 in range(bit2 + 1, N):
                    for bit4 in range(bit3 + 1, N):
                        for bit5 in range(bit4 + 1, N):
                            neighbors.append(w ^ (1 << bit1) ^ (1 << bit2) ^ (1 << bit3) ^ (1 << bit4) ^ (1 << bit5))
        return neighbors
    
    # This is too slow to call repeatedly. Let's precompute a lookup table.
    # For each word, store a bitmask of its neighbors? No, 9401 bits per word is too much.
    
    # Alternative: use the all_vectors array to compute distances vectorized.
    # The current approach does this. Let's optimize it.
    
    # Key optimization: instead of checking all candidates against all codewords,
    # we can check all candidates against one codeword at a time, and eliminate 
    # candidates that conflict with any codeword.
    
    # Let's try a different structural approach: 
    #   - Use a "maximal code" construction via the Lovasz local lemma style.
    #   - Or, use a genetic algorithm / simulated annealing on the code.
    
    # Given the time constraints, let's stick with random greedy but with these improvements:
    #   1. Use a larger batch size for candidate evaluation.
    #   2. Use multiple independent restarts with different random seeds.
    #   3. After each greedy pass, try a local search to improve the code.
    
    # Let's implement a more efficient version of random greedy:
    #   - Maintain a list of available candidates.
    #   - At each step, sample a random subset of candidates.
    #   - For each candidate in the subset, check if it's valid (min dist >= 6).
    #   - Add all valid candidates from the subset.
    #   - Repeat until no more candidates can be added.
    
    # This is essentially what the current code does. Let's add a local search step.
    
    # Actually, let's try a completely different structural change:
    #   - Use a "spherical code" perspective: map binary words to points on a sphere,
    #     and use a repulsion-based algorithm to spread them out.
    #   - This is a different optimization family.
    
    # But that might be too complex. Let's try a simpler structural change:
    #   - Use a "greedy with lookahead": when choosing the next word, consider 
    #     not just its distance to the current code, but also how many future 
    #     candidates it would block.
    
    # Given the complexity, let's implement a clean version of random greedy 
    # with better efficiency and more restarts.
    
    def greedy_pass(seed_idx_list, deadline):
        """Perform a random greedy pass starting from the given seed."""
        current_indices = list(seed_idx_list)
        current_set = set(current_indices)
        
        # Available candidates: all words not in current code
        available = np.setdiff1d(all_indices, current_indices)
        np.random.shuffle(available)
        
        # We'll process candidates in order, adding those that are valid.
        # To speed up, we'll use a vectorized distance check.
        # Maintain an array of current codeword vectors.
        current_vecs = all_vectors[np.array(current_indices)]
        
        added = 0
        for idx in available:
            if time.time() > deadline:
                break
            c_vec = all_vectors[idx]
            # Compute distances to all current codewords
            dists = np.sum(current_vecs != c_vec, axis=1)
            if np.min(dists) >= MIN_DIST:
                current_indices.append(idx)
                current_set.add(idx)
                current_vecs = np.vstack([current_vecs, c_vec])
                added += 1
        
        return current_indices
    
    # Run multiple greedy passes with different random seeds
    for restart in range(100):
        if time.time() > deadline - 2:
            break
        
        # Shuffle the seed order to get different starting points
        shuffled_seed = list(seed_indices)
        np.random.shuffle(shuffled_seed)
        
        current_code = greedy_pass(shuffled_seed, deadline - 1)
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
    
    return best_code
