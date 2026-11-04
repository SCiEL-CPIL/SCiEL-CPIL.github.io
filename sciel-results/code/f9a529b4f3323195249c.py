"""Improved ternary covering code K_3(7,1) via greedy + local search."""
import time
import itertools
import numpy as np

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a greedy construction with local search to minimize codeword count.
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: use tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # For q=3, n=7, radius=1:
    # Each codeword covers 1 + 2*7 = 15 words (itself + 14 neighbors)
    # Total space size: 3^7 = 2187
    # Lower bound: ceil(2187/15) = 146, but actual lower bound is 156
    # Best known: 186
    
    # Greedy approach:
    # 1. Start with empty set of codewords
    # 2. Repeatedly add the codeword that covers the most uncovered words
    # 3. Local search: try to remove codewords and see if coverage is maintained
    # 4. Kick: randomly remove some codewords and re-greedy
    
    # Precompute all words as indices 0..2186
    # Word i in base 3: digits d0..d6 where i = d0*3^6 + d1*3^5 + ... + d6*3^0
    
    # For each word, precompute its neighbors (words at distance <= 1)
    # A word has 14 neighbors at distance 1 (7 positions * 2 other values)
    
    # Efficient representation:
    # For each word w, neighbors(w) = {w} union {w with one digit changed}
    
    # Let's use numpy for vectorized operations
    # Represent all 2187 words as a 2187x7 array of digits
    
    all_words = np.zeros((3**n, n), dtype=np.int8)
    for idx in range(3**n):
        temp = idx
        for pos in range(n-1, -1, -1):
            all_words[idx, pos] = temp % 3
            temp //= 3
    
    # For each word, compute the set of words it covers (itself + neighbors)
    # Instead of storing sets, we can use a coverage counter approach
    
    # Greedy construction:
    # covered = boolean array of size 2187
    # For each candidate codeword c, gain(c) = number of uncovered words in neighbors(c)
    # Pick c with max gain, add to codewords, update covered
    
    # To speed up, precompute for each word w, the list of words it covers
    # cover_set[w] = list of indices of words within distance 1 of w
    
    cover_sets = []
    for w in range(3**n):
        neighbors = [w]
        for pos in range(n):
            for val in range(3):
                if val != all_words[w, pos]:
                    # Create new word with pos changed to val
                    new_idx = w
                    # Compute new index
                    # w's digit at pos is all_words[w, pos]
                    # new digit is val
                    # difference: (val - all_words[w, pos]) * 3^(n-1-pos)
                    diff = (val - int(all_words[w, pos])) * (3 ** (n - 1 - pos))
                    new_idx = w + diff
                    neighbors.append(new_idx)
        cover_sets.append(neighbors)
    
    # Greedy phase
    covered = np.zeros(3**n, dtype=bool)
    codewords = []
    
    while not covered.all():
        if time.time() > deadline - 30:
            break
        
        # Find the word with maximum coverage of uncovered words
        # gain[w] = sum of (not covered[x]) for x in cover_sets[w]
        # This is expensive to compute naively. Let's use a smarter approach.
        
        # Alternative: maintain for each uncovered word, which codewords could cover it
        # But that's complex. Let's just compute gains periodically.
        
        # For efficiency, we'll sample a subset of candidates or use a heuristic
        # Actually, let's just compute all gains - 2187 * 15 = 32805 operations per iteration
        # With ~186 iterations, that's ~6 million ops, should be fine in numpy
        
        gains = np.zeros(3**n, dtype=np.int32)
        for w in range(3**n):
            gains[w] = np.sum(~covered[cover_sets[w]])
        
        # Pick word with max gain
        best_w = np.argmax(gains)
        if gains[best_w] == 0:
            break
        
        codewords.append(int(best_w))
        for x in cover_sets[best_w]:
            covered[x] = True
    
    # Local search: try to remove codewords
    # For each codeword, check if removing it still leaves all words covered
    # A word is still covered if there's another codeword within distance 1
    
    def is_valid_cover(codeword_list):
        if not codeword_list:
            return False
        # Check if every word is within distance 1 of some codeword
        covered_local = np.zeros(3**n, dtype=bool)
        for cw in codeword_list:
            for x in cover_sets[cw]:
                covered_local[x] = True
        return covered_local.all()
    
    # Try removing codewords one at a time
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(len(codewords)):
            # Try removing codeword i
            new_codewords = codewords[:i] + codewords[i+1:]
            if is_valid_cover(new_codewords):
                codewords = new_codewords
                improved = True
                break  # Restart from beginning after removal
    
    # Kick phase: randomly remove some codewords and re-greedy
    for kick_iter in range(5):
        if time.time() > deadline - 10:
            break
        
        # Remove a random subset of codewords
        k = min(3, len(codewords) // 5)
        if k <= 0:
            break
        
        to_remove = set(np.random.choice(len(codewords), k, replace=False))
        kicked_codewords = [cw for i, cw in enumerate(codewords) if i not in to_remove]
        
        # Re-greedy to fill gaps
        covered_local = np.zeros(3**n, dtype=bool)
        for cw in kicked_codewords:
            for x in cover_sets[cw]:
                covered_local[x] = True
        
        new_codewords = list(kicked_codewords)
        while not covered_local.all() and time.time() < deadline - 10:
            gains = np.zeros(3**n, dtype=np.int32)
            for w in range(3**n):
                gains[w] = np.sum(~covered_local[cover_sets[w]])
            
            best_w = np.argmax(gains)
            if gains[best_w] == 0:
                break
            
            new_codewords.append(int(best_w))
            for x in cover_sets[best_w]:
                covered_local[x] = True
        
        # Local search on new_codewords
        improved2 = True
        while improved2 and time.time() < deadline - 10:
            improved2 = False
            for i in range(len(new_codewords)):
                candidate = new_codewords[:i] + new_codewords[i+1:]
                if is_valid_cover(candidate):
                    new_codewords = candidate
                    improved2 = True
                    break
        
        if len(new_codewords) < len(codewords):
            codewords = new_codewords
    
    # Convert back to list of lists
    result = [list(all_words[cw]) for cw in codewords]
    
    # Ensure validity
    if not is_valid_cover(codewords):
        # Fallback to tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return result
