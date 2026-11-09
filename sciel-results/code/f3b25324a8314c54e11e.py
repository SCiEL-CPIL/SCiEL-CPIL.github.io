import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    timeout = 140  # Safety margin from 150s limit
    
    # Fallback for non-target problems
    if q != 3 or n != 7 or radius != 1:
        # Simple product construction for general case
        # Find smallest m such that we can split n into parts
        # For now, use a basic greedy
        all_words = list(itertools.product(range(q), repeat=n))
        # Return all words as fallback (correct but not minimal)
        return [list(w) for w in all_words]
    
    # Optimized for q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    
    # Precompute digits array: shape (2187, 7), LSB first
    digits_array = np.zeros((total_words, n), dtype=np.int32)
    for i in range(total_words):
        temp = i
        for k in range(n):
            digits_array[i, k] = temp % 3
            temp //= 3
    
    # Precompute ball membership: for each word, list of words within distance 1
    # Ball size = 1 + 7*2 = 15
    ball_indices = np.zeros((total_words, 15), dtype=np.int32)
    for i in range(total_words):
        ball_indices[i, 0] = i
        idx = 1
        for pos in range(n):
            orig = digits_array[i, pos]
            p = powers3[pos]
            base = i - orig * p
            for nd in range(3):
                if nd == orig:
                    continue
                ball_indices[i, idx] = base + nd * p
                idx += 1

    # Helper function to check if a set of codewords covers all words
    def is_covering(code_ints):
        if not code_ints:
            return False
        covered = np.zeros(total_words, dtype=bool)
        for c in code_ints:
            covered[ball_indices[c]] = True
        return np.all(covered)

    # Helper function to perform local search (Delete step)
    def local_search(code_ints):
        code_set = set(code_ints)
        improved = True
        while improved:
            improved = False
            # Try to remove each codeword
            for c in list(code_set):
                # Check if removing c leaves any word uncovered
                # A word w is uncovered if it is in the ball of c 
                # and not in the ball of any other c' in code_set \ {c}
                
                # Efficient check:
                # For each word w in ball(c), check if it is covered by others.
                # If all w in ball(c) are covered by others, c is redundant.
                
                redundant = True
                for w_idx in range(15):
                    w = ball_indices[c, w_idx]
                    # Check if w is covered by any other codeword
                    # We can check if any other codeword covers w.
                    # Instead of iterating all others, we can maintain a count array.
                    # But for simplicity and speed in Python, let's just check.
                    # Actually, iterating all others is O(|C| * 15). 
                    # Let's use a count array approach for the whole local search loop?
                    # No, let's just do the direct check for now, it's fast enough for |C|~200.
                    
                    covered_by_other = False
                    for c2 in code_set:
                        if c2 == c:
                            continue
                        # Check if w is in ball(c2)
                        # We can precompute a reverse mapping? 
                        # Or just check if dist(w, c2) <= 1.
                        # dist(w, c2) = sum(digits_array[w] != digits_array[c2])
                        # This is O(n) per pair. 
                        # 15 * 200 * 7 = 21000 ops per deletion check.
                        # 200 deletions * 21000 = 4.2M ops per pass.
                        # This is acceptable.
                        
                        if np.sum(digits_array[w] != digits_array[c2]) <= 1:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                
                if redundant:
                    code_set.remove(c)
                    improved = True
                    break # Restart loop as set changed
            
        return list(code_set)

    best_code = None
    best_size = total_words + 1
    
    rng = np.random.default_rng(12345)
    
    # Try different strategies
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        uncovered = np.ones(total_words, dtype=bool)
        code_ints = []
        
        while np.any(uncovered):
            # Compute coverage: for each candidate, how many uncovered words it covers
            # coverage[i] = sum(uncovered[ball_indices[i]])
            # Vectorized: for each i, sum uncovered over its ball
            coverage = np.zeros(total_words, dtype=np.int32)
            for j in range(15):
                coverage += uncovered[ball_indices[:, j]]
            
            # Find best candidate
            max_cov = np.max(coverage)
            if max_cov == 0:
                break
            
            candidates = np.where(coverage == max_cov)[0]
            
            # Randomize tie-breaking for diversity
            if len(candidates) > 1 and r > 0:
                # Pick from top candidates with some randomness
                if len(candidates) > 10:
                    # Pick randomly from top 10
                    top_idx = np.argsort(-coverage)[:10]
                    best_idx = rng.choice(top_idx)
                else:
                    best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
            
            code_ints.append(best_idx)
            
            # Update uncovered
            uncovered[ball_indices[best_idx]] = False
        
        # Apply local search polish
        polished_code = local_search(code_ints)
        
        if len(polished_code) < best_size:
            best_size = len(polished_code)
            best_code = polished_code.copy()
            
            # If we've found a very good solution, we can stop early
            if best_size <= 200:
                # Try a few more restarts to see if we can improve
                continue
    
    if best_code is None:
        # Fallback to product construction: 243 words
        # Ternary Hamming code of length 4
        hamming_code = []
        for a in range(3):
            for b in range(3):
                c = (-a - 2*b) % 3  # a + b + 2c = 0 mod 3? Let me verify
                # Actually, the standard ternary Hamming code of length 4:
                # Parity check matrix H = [[1,0,1,1],[0,1,1,2]]
                # Codewords satisfy H*x = 0 mod 3
                # x0 + x2 + x3 = 0
                # x1 + x2 + 2*x3 = 0
                # Let x0=a, x1=b, then x2 = -a-b-2x3... this is getting complicated
                # Use known 9 codewords
                pass
        
        # Simpler: use the known 9-word perfect code
        # The ternary Hamming code [4,2,3]_3 has 9 codewords
        # Generate: all (a,b,a+b,a+2b) mod 3? Let me verify coverage
        # Actually, let's just use the product construction directly
        # C4 = {(a, b, (a+b)%3, (a+2*b)%3) for a,b in 0..2}
        c4 = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        
        # Convert to integers
        def word_to_int(w):
            result = 0
            for k, d in enumerate(w):
                result += d * powers3[k]
            return result
        
        code4_ints = [word_to_int(c) for c in c4]
        
        # Product with all length-3 words
        result = []
        for tail in itertools.product(range(3), repeat=3):
            tail_int = tail[0] * 1 + tail[1] * 3 + tail[2] * 9
            for c_int in code4_ints:
                # Combine: first 4 digits from c_int, last 3 from tail_int
                # c_int has digits at positions 0-3, tail_int at positions 0-2
                # Combined: c_int + tail_int * 81 (since 3^4 = 81)
                combined = c_int + tail_int * 81
                result.append(combined)
        
        best_code = result
        best_size = len(result)
    
    # Convert to list of lists
    result = []
    for i in best_code:
        word = []
        temp = i
        for k in range(n):
            word.append(temp % 3)
            temp //= 3
        result.append(word)
    
    return result
