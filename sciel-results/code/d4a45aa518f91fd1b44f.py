import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Seed: First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # This gives 32 words of length 17 with distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The 17th bit (index 16) is 0 for both extensions? 
            # The task says "extended by two zero bits" in the comment, but N=17.
            # The seed code uses 16 points. The 17th bit is implicitly 0.
            words.append(word)
    
    code = set(words)
    N = 17
    MIN_DIST = 6
    
    # Precompute all candidates
    all_candidates = list(range(1 << N))
    
    # Initialize min distances
    # min_dists[i] is the min distance between candidate i and any word in code
    min_dists = [100] * len(all_candidates) # 100 is > max possible dist (17)
    
    # Calculate initial min distances
    # This is the most expensive part initially
    # We can optimize by breaking early if dist < MIN_DIST
    for i, c in enumerate(all_candidates):
        if c in code:
            min_dists[i] = 0
            continue
        min_d = 100
        for w in code:
            d = (c ^ w).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    break
        min_dists[i] = min_d

    # Greedy loop
    while time.time() < deadline:
        # Find the candidate with the maximum min distance that is still valid (>= MIN_DIST)
        best_idx = -1
        best_d = -1
        for i in range(len(all_candidates)):
            if min_dists[i] >= MIN_DIST and min_dists[i] > best_d:
                best_d = min_dists[i]
                best_idx = i
        
        if best_idx == -1:
            break
        
        new_word = all_candidates[best_idx]
        code.add(new_word)
        min_dists[best_idx] = 0 # Mark as used/invalid
        
        # Update min distances for all other candidates
        # We only need to update if the new distance is smaller
        for i in range(len(all_candidates)):
            if min_dists[i] < MIN_DIST:
                continue # Already invalid
            d = (all_candidates[i] ^ new_word).bit_count()
            if d < min_dists[i]:
                min_dists[i] = d
                # If it drops below MIN_DIST, it becomes invalid for future selection
                # We don't remove it, just mark min_dists[i] < MIN_DIST
        
    return list(code)
