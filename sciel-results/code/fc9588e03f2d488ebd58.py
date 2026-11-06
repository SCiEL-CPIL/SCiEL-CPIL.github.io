import numpy as np
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Start with Reed-Muller [16,5,8] code extended by 2 zero bits -> 32 words, dist 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    code = list(base_words)
    code_set = set(code)
    
    # Precompute popcount lookup for 18-bit values using a dictionary or array
    # Actually, let's use a faster approach: precompute popcount for all 2^18 values
    popcount = np.zeros(2**18, dtype=np.uint8)
    for i in range(2**18):
        popcount[i] = i.bit_count()
    
    # Convert code to numpy array for faster access
    code_np = np.array(code, dtype=np.uint32)
    
    # Greedy addition with better candidate ordering
    # Strategy: try candidates in order of increasing weight, which tends to find
    # valid codewords earlier and allows us to build a denser code
    
    # Generate candidates by weight class
    # For each weight w from 0 to 18, generate all words of that weight
    # This is expensive, so let's use a smarter approach
    
    # Better: use a priority queue-like approach where we track which candidates
    # are "promising" based on their distance to existing codewords
    
    # Simple improvement: iterate through candidates in a structured order
    # that tends to find valid codewords quickly
    
    # Let's try multiple passes with different orderings
    # Pass 1: increasing weight
    # Pass 2: random order
    
    # First, let's do a quick greedy pass with weight-ordered candidates
    # For each weight, we can generate combinations
    
    from itertools import combinations
    
    # Generate all words by weight
    # This is memory intensive, so let's be selective
    # Only try weights that are likely to be useful
    
    # Actually, let's just use a faster distance check and iterate through
    # all candidates in a better order
    
    # Key insight: if we check candidates in order of their "distance potential"
    # (i.e., how far they are from existing codewords), we'll find valid ones faster
    
    # For now, let's use a simple improvement: check candidates in order of
    # increasing weight, which often works well for error-correcting codes
    
    # Generate candidates in weight order
    candidates_by_weight = {}
    for w in range(0, N+1):
        if w <= 9:  # Only generate up to weight 9 to save time
            candidates_by_weight[w] = []
            for combo in combinations(range(N), w):
                word = 0
                for bit_pos in combo:
                    word |= (1 << bit_pos)
                if word not in code_set:
                    candidates_by_weight[w].append(word)
    
    # Try adding candidates in weight order
    for w in range(0, N+1):
        if time.time() > deadline:
            break
        
        if w not in candidates_by_weight:
            continue
        
        for candidate in candidates_by_weight[w]:
            if time.time() > deadline:
                break
            
            if candidate in code_set:
                continue
            
            # Fast distance check using precomputed popcount
            min_d = N  # max possible
            for cw in code:
                d = popcount[candidate ^ cw]
                if d < min_d:
                    min_d = d
                if min_d < MIN_DIST:
                    break
            
            if min_d >= MIN_DIST:
                code.append(candidate)
                code_set.add(candidate)
    
    # Now do a random pass to find more codewords
    rng = np.random.default_rng(42)
    
    # Generate random candidates
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        candidate = int(rng.integers(0, 2**18))
        if candidate in code_set:
            continue
        
        # Fast distance check
        min_d = N
        for cw in code:
            d = popcount[candidate ^ cw]
            if d < min_d:
                min_d = d
            if min_d < MIN_DIST:
                break
        
        if min_d >= MIN_DIST:
            code.append(candidate)
            code_set.add(candidate)
    
    return code
