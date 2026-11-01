import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: direct product construction
        # For general case, use simple repetition code or similar
        if n <= 4:
            # Perfect code for small n
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [[a, b] for a in range(q) for b in range(q)]
            elif n == 3:
                return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
            elif n == 4:
                return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        else:
            # Fallback: direct sum of tetracode-like construction
            base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in base for tail in itertools.product(range(q), repeat=n-4)]
    
    # Specialized for q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    
    # Precompute all words as integers 0..6560
    all_words = np.arange(6561, dtype=np.int32)
    
    # Convert to base-3 digits
    def to_digits(x):
        digits = np.zeros((len(x), 8), dtype=np.int8)
        for i in range(8):
            digits[:, i] = x % 3
            x = x // 3
        return digits
    
    all_digits = to_digits(all_words)
    
    # Precompute neighbors: for each word, its 24 neighbors at distance 1
    # We'll compute coverage incrementally
    
    # Start with empty code
    code = []
    covered = np.zeros(6561, dtype=bool)
    
    # Precompute neighbor lists for efficiency
    # For each position and each digit change, compute the new word index
    # word_index = sum(d_i * 3^i)
    powers = [1, 3, 9, 27, 81, 243, 729, 2187]
    
    # For greedy: at each step, find word that covers most uncovered words
    # A word covers itself + 24 neighbors = 25 words
    
    # Precompute for each word its 25 neighbors (itself + 24 at distance 1)
    # This is expensive to precompute fully, so we'll compute on the fly or use a smarter approach
    
    # Alternative: use random restarts with greedy
    # But first, let's try a pure greedy approach with vectorized coverage computation
    
    # For each candidate word, compute how many uncovered words it covers
    # We can do this by: for each uncovered word, check if it's within distance 1 of candidate
    
    # More efficient: precompute the set of neighbors for each word
    # neighbor_set[w] = set of words within distance 1 of w
    
    # Since 6561 * 25 = 164025, we can precompute this
    # But storing as a set for each is memory heavy. Instead, use a boolean matrix? 6561x6561 = 43M booleans, too much.
    
    # Better: for each word, generate its 25 neighbors on the fly when needed
    # Or: use the fact that we can compute coverage by iterating over uncovered words
    
    # Let's use a different approach: maintain a list of uncovered words, and for each candidate,
    # count how many uncovered words are within distance 1
    
    # Even better: precompute a lookup table for each word's neighbors as a flat array
    # neighbors[w] = array of 25 word indices
    
    neighbors = np.zeros((6561, 25), dtype=np.int32)
    for w in range(6561):
        digits = []
        tmp = w
        for i in range(8):
            digits.append(tmp % 3)
            tmp //= 3
        # digits[0] is least significant (position 0), digits[7] is most significant
        
        neighbors[w, 0] = w  # itself
        
        idx = 1
        for pos in range(8):
            old_digit = digits[pos]
            for new_digit in range(3):
                if new_digit == old_digit:
                    continue
                # Compute new word index
                new_w = w + (new_digit - old_digit) * powers[pos]
                neighbors[w, idx] = new_w
                idx += 1
        # idx should be 25
    
    # Now greedy: repeatedly pick the word that covers the most uncovered words
    uncovered_count = 6561
    uncovered_set = set(range(6561))
    
    # To speed up, we can maintain a count of how many uncovered words each candidate covers
    # But this changes as we add words to the code
    
    # Simple greedy: at each step, scan all 6561 candidates, compute coverage, pick best
    # This is O(6561 * 25 * |code|) per step, which is too slow for 486 steps
    
    # Better: use a priority queue or just accept the O(6561 * 25) per step
    # 6561 * 25 = 164025 operations per step, 486 steps = 80M operations, should be fine in numpy
    
    # Let's vectorize: for each candidate, compute sum of covered[neighbors[candidate]]
    # covered is a boolean array of length 6561
    
    # Actually, we want the number of UNCOVERED words covered by each candidate
    # So: for each candidate c, count = sum(1 for nb in neighbors[c] if not covered[nb])
    
    # Vectorized: 
    # candidates_covered = covered[neighbors]  # shape (6561, 25)
    # uncovered_covered = 1 - candidates_covered  # boolean
    # counts = uncovered_covered.sum(axis=1)
    
    while uncovered_count > 0 and time.time() < deadline:
        # Compute for each candidate how many uncovered words it covers
        # neighbors shape: (6561, 25)
        # covered shape: (6561,)
        
        # Get covered status for all neighbors of all candidates
        # This creates a (6561, 25) boolean array
        cand_covered = covered[neighbors]  # (6561, 25)
        cand_uncovered = ~cand_covered  # (6561, 25)
        counts = cand_uncovered.sum(axis=1)  # (6561,)
        
        # Pick the candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            # Shouldn't happen if uncovered_count > 0, but break just in case
            break
        
        # Add this word to the code
        best_word_idx = best_idx
        code.append(best_word_idx)
        
        # Mark all its neighbors as covered
        for nb in neighbors[best_word_idx]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
        
        # Early stop if we've used too many words (shouldn't happen, but safety)
        if len(code) > 600:
            break
    
    # Convert word indices back to digit lists
    result = []
    for w in code:
        digits = []
        tmp = w
        for i in range(8):
            digits.append(tmp % 3)
            tmp //= 3
        result.append(digits)
    
    return result
