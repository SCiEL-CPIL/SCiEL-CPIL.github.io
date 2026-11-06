import random
import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 45.0  # Safety margin for 60s limit

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    # Build the 256-word shortened quadratic residue code as a base set.
    # The [18,9,6] extended quadratic-residue code over GF(2) has a generator matrix
    # based on the quadratic residues of 17: {1,2,4,8,9,13,15,16}.
    # We construct the [17,9,6] code by shortening one coordinate.
    
    # Quadratic residues mod 17
    QR = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator matrix for the [18,9,6] code
    # The extended QR code has a generator matrix where row i corresponds to
    # a cyclic shift. Let's construct it properly.
    
    # Actually, let's use a known construction. The [17,5,9] Hamming code is too small.
    # Let me use a different approach: start with a good structured code.
    
    # Use Reed-Muller RM(1,4) extended to 17 bits as base (32 words, dist 8)
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                dot = bin(slope & point).count('1') % 2
                bit = (dot + intercept) % 2
                word |= bit << point
            base_words.append(word)  # bit 16 is 0
    
    codewords = list(base_words)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Precompute compatibility with base words for speed
    # For a candidate word w, check if it's compatible with all base words
    
    rng = random.Random(42)
    
    def is_compatible_with_base(w):
        for c in base_words:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
    
    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
    
    # Phase 1: Greedy addition of random compatible words
    attempts = 0
    max_attempts_greedy = 200000
    while attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.3:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Phase 2: Try to find more words by considering candidates near existing codewords
    # and using a local search approach
    
    # Phase 3: Use a more structured search - try all weight-8 and weight-9 words
    # that are compatible with the current set
    
    # First, let's try to add more words using a smarter candidate generation
    # Focus on words that are at distance >= 6 from all existing codewords
    
    # Build a set of candidates by flipping bits in existing codewords
    candidate_set = set()
    for c in codewords:
        # Try flipping 1-3 bits
        for mask in range(1, 1 << N):
            if bin(mask).count('1') > 3:
                continue
            w = c ^ mask
            if w not in candidate_set:
                candidate_set.add(w)
    
    # Also add some random candidates
    for _ in range(5000):
        candidate_set.add(rng.getrandbits(N))
    
    # Filter candidates that are compatible with all current codewords
    valid_candidates = []
    for w in candidate_set:
        if is_compatible_with_all(w, codewords):
            valid_candidates.append(w)
    
    # Now find a large clique among valid_candidates
    # Use a greedy clique heuristic
    
    def greedy_clique(candidates, existing):
        """Find a large clique in the compatibility graph of candidates,
        where all candidates must also be compatible with existing words."""
        if not candidates:
            return []
        
        clique = []
        remaining = list(candidates)
        
        # Sort by degree (number of compatible other candidates)
        def degree(c):
            count = 0
            for other in remaining:
                if other != c and hamming_dist(c, other) >= MIN_DIST:
                    count += 1
            return count
        
        # Greedy: repeatedly pick the vertex with highest degree
        while remaining:
            if time.time() - start_time > time_limit * 0.8:
                break
            
            # Find vertex with highest degree
            best_idx = 0
            best_deg = -1
            for i, c in enumerate(remaining):
                deg = 0
                for other in remaining:
                    if other != c and hamming_dist(c, other) >= MIN_DIST:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_idx = i
            
            c = remaining.pop(best_idx)
            clique.append(c)
            
            # Remove all candidates incompatible with c
            remaining = [other for other in remaining 
                        if hamming_dist(c, other) >= MIN_DIST]
        
        return clique
    
    # Apply greedy clique to valid_candidates
    if valid_candidates and time.time() - start_time < time_limit * 0.7:
        # Limit the number of candidates to avoid O(n^2) blowup
        if len(valid_candidates) > 5000:
            # Sample a subset
            valid_candidates = rng.sample(valid_candidates, 5000)
        
        clique = greedy_clique(valid_candidates, codewords)
        
        if len(clique) > 0:
            # Verify all clique members are compatible with existing codewords
            for w in clique:
                if not is_compatible_with_all(w, codewords):
                    clique = []
                    break
            
            if clique:
                codewords.extend(clique)
                if len(codewords) > best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
    
    # Phase 4: Final greedy polish
    attempts = 0
    max_attempts_final = 100000
    while attempts < max_attempts_final:
        if time.time() - start_time > time_limit:
            break
        
        w = rng.getrandbits(N)
        if is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Return the best set found
    return best_codewords
