import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    total_words = q ** n
    
    # Precompute neighbors (ball of radius 1) for each word
    # Using a list of lists of neighbor indices
    balls = []
    # Precompute the ball mask for each word as a Python int (bitset)
    ball_masks = []
    
    # Helper: index <-> word conversion
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Build balls and masks
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_idx(w))
            w[pos] = orig
        balls.append(neighbors)
        
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
    
    all_mask = (1 << total_words) - 1
    
    # Optimization: Precompute for each candidate c, the list of words it covers
    # This is already 'balls[c]'
    
    # Greedy cover with tie-breaking by "efficiency" or random choice
    def greedy_cover(seed, time_limit):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            # To speed up, we can skip candidates that are already in 'code'
            # but their gain would be 0 anyway.
            # However, checking 'c in code' is O(N) for a list.
            # Since N is ~186 and total_words is 2187, it's fine.
            # But we can just compute gain. If gain is 0, it won't be selected unless all are 0.
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > time_limit:
                return None
        
        return code
    
    def reduce_cover(code, time_limit):
        if not code:
            return code
        
        # Count how many times each word is covered
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow popping
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
            
            if time.time() > time_limit:
                break
        
        return code
    
    def iterated_greedy(seed, time_limit):
        random.seed(seed)
        
        # 1. Initial Greedy
        code = greedy_cover(seed, time_limit)
        if code is None:
            return None
        
        # 2. Reduce
        code = reduce_cover(code, time_limit)
        current_size = len(code)
        
        # 3. Perturbation & Repair Loop
        # We try to remove a small number of codewords and repair.
        # This is a form of local search (k-opt).
        
        for round_num in range(20):  # Max 20 perturbation rounds
            if time.time() > time_limit:
                break
            
            # Choose k to remove. k=1 or k=2.
            # Larger k explores more but is harder to repair.
            k = 1 if random.random() < 0.7 else 2
            
            if len(code) <= k:
                break
            
            # Select indices to remove
            # To make it more effective, we might remove codewords that cover "less unique" words?
            # But random is a good baseline.
            indices_to_remove = random.sample(range(len(code)), k)
            remove_set = set(indices_to_remove)
            
            # Create new code list
            new_code = [c for i, c in enumerate(code) if i not in remove_set]
            
            # Calculate current coverage of new_code
            covered_mask = 0
            for c in new_code:
                covered_mask |= ball_masks[c]
            
            # Repair: Greedily add codewords to cover holes
            # We need to cover (all_mask ^ covered_mask)
            
            while covered_mask != all_mask:
                best_gain = -1
                best_candidates = []
                
                # Optimization: Only consider candidates that cover at least one hole.
                # If gain is 0, skip.
                for c in range(total_words):
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [c]
                    elif gain == best_gain:
                        best_candidates.append(c)
                
                if not best_candidates:
                    break
                
                chosen = random.choice(best_candidates)
                new_code.append(chosen)
                covered_mask |= ball_masks[chosen]
                
                if time.time() > time_limit:
                    return None
            
            # Reduce the repaired code
            new_code = reduce_cover(new_code, time_limit)
            
            # Accept if better
            if len(new_code) < current_size:
                code = new_code
                current_size = len(new_code)
            elif len(new_code) == current_size:
                # Accept with probability to explore (Simulated Annealing style, but simple)
                # If different, accept to allow further moves
                if code != new_code:
                    if random.random() < 0.5:
                        code = new_code
        
        return code
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple iterations with different seeds
    # We need to be careful with time.
    # Let's run a quick greedy first to get a baseline.
    
    # Initial greedy
    init_code = greedy_cover(0, deadline)
    if init_code:
        reduced = reduce_cover(init_code, deadline)
        best_solution = reduced
        best_size = len(reduced)
    
    # Now run iterated greedy
    seed = 1
    while time.time() < deadline - 5:  # Keep 5s buffer
        # Estimate time per iteration?
        # Just run one.
        code = iterated_greedy(seed, deadline - 2)
        if code is None:
            break
        
        if len(code) < best_size:
            best_solution = code
            best_size = len(code)
        
        seed += 1
        
    if best_solution is None:
        # Fallback: Return a valid (though large) code
        # e.g. all words? No, that's 2187.
        # A simple construction:
        # For each position, add words that cover that position?
        # Actually, the greedy above should always find something.
        # If it fails, it's a bug or timeout.
        # Let's return a single word? No, must cover all.
        # Return all words?
        return [idx_to_word(i) for i in range(total_words)]
    
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
