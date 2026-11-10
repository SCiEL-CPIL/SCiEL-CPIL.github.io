import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # The evaluator passes q, n, radius.
    
    N = q ** n
    start_time = time.time()
    # Safety margin: aim to finish by 115s, hard kill at 120s
    deadline = start_time + 115.0
    
    # Precompute neighbors (balls) for all words
    # We represent words as integers 0..N-1
    # idx_to_digits and digits_to_idx helpers
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
        
    def digits_to_idx(digits):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + digits[i]
        return idx

    # Precompute the ball for each word
    # ball[i] is a list of indices j such that dist(i, j) <= radius
    balls = [None] * N
    for i in range(N):
        d = idx_to_digits(i)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = d[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    d[pos] = new_val
                    nb_idx = digits_to_idx(d)
                    ball.append(nb_idx)
                d[pos] = orig_val
        balls[i] = ball
        
    # Precompute a coverage mask matrix for fast vectorized operations
    # cover_mask[i] is a boolean array of length N, where cover_mask[i][j] is True if j is in balls[i]
    # Since N=2187, this is 2187 * 2187 * 1 byte ~ 4.5 MB, which is very manageable.
    cover_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        cover_mask[i, balls[i]] = True

    def is_valid(codewords):
        if not codewords:
            return N == 0
        # Check if union of balls covers all words
        # sum of cover_mask[codewords, :] along axis 0 gives count of codewords covering each word
        # We just need to know if any word has count 0
        # Optimization: if len(codewords) is small, we can just OR the rows
        # But for ~180 codewords, summing is fast with numpy
        coverage = cover_mask[codewords, :].sum(axis=0)
        return np.all(coverage >= 1)

    def greedy_cover():
        """Standard greedy set cover to get a valid initial solution."""
        covered = np.zeros(N, dtype=bool)
        code = []
        
        while True:
            if time.time() > deadline - 10:
                break
            if np.all(covered):
                break
            
            # Calculate gain: number of currently uncovered words covered by each candidate
            # uncovered_mask = ~covered
            # gains = sum(cover_mask * uncovered_mask, axis=1)
            # Using dot product with boolean cast is efficient
            uncovered_mask = ~covered
            # Only consider candidates that cover at least 1 uncovered word? 
            # Actually, just find the one with max gain.
            gains = cover_mask.dot(uncovered_mask.astype(np.uint8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if space is connected and radius >= 0
                break
                
            # Find all with max gain, pick randomly to diversify
            candidates = np.where(gains == max_gain)[0]
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(int(best_c))
            covered[balls[best_c]] = True
            
        return code

    def local_search_reduce(codewords):
        """
        Iteratively remove redundant codewords.
        A codeword c is redundant if for all w in balls[c], there exists another c' in code such that w in balls[c'].
        Equivalent to: min(coverage_count[balls[c]]) > 1.
        """
        if not codewords:
            return codewords
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Compute coverage counts
            # cover_mask[codewords, :] is (K, N). Sum along axis 0 -> (N,)
            coverage_count = cover_mask[codewords, :].sum(axis=0)
            
            # Check each codeword for redundancy
            # To make this faster, we can check in random order or just iterate.
            # If we remove one, we must recompute coverage_count for the next iteration.
            # Is there a faster way? 
            # If we remove c, coverage_count[balls[c]] decreases by 1.
            # We can maintain coverage_count incrementally.
            
            # Let's try incremental update for speed.
            # But for simplicity and robustness in this short window, recompute is O(K*N).
            # K~180, N=2187. 180*2187 ~ 400k ops. Very fast.
            
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if min coverage in its ball is > 1
                # If so, it is redundant.
                if np.min(coverage_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop to update coverage_count efficiently
            
        return codewords

    def repair(codewords):
        """
        If codewords is invalid, add codewords to make it valid.
        Uses greedy strategy to add the best covering codeword for uncovered words.
        """
        if is_valid(codewords):
            return codewords
        
        # Compute current coverage
        coverage_count = np.zeros(N, dtype=np.int32)
        for c in codewords:
            coverage_count[balls[c]] += 1
            
        while True:
            if time.time() > deadline - 2:
                break
                
            uncovered_mask = (coverage_count == 0)
            if not np.any(uncovered_mask):
                break
                
            # Find best candidate to cover uncovered words
            # Gain: number of uncovered words covered by candidate
            gains = cover_mask.dot(uncovered_mask.astype(np.uint8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Stuck, shouldn't happen
                
            candidates = np.where(gains == max_gain)[0]
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            
            codewords.append(int(best_c))
            coverage_count[balls[best_c]] += 1
            
        return codewords

    # 1. Initial Solution
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code)
    best_len = len(best_code)
    
    current_code = best_code[:]
    
    # 2. Iterated Local Search
    # We will perturb, repair, reduce, and accept based on SA criteria
    
    T = 5.0 # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline - 5:
        # Perturbation
        child = current_code[:]
        
        # Remove 1 to 3 codewords
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if not child: break
            idx = random.randint(0, len(child) - 1)
            child.pop(idx)
        
        # Add 1 to 3 random codewords
        num_add = random.randint(1, 3)
        for _ in range(num_add):
            new_c = random.randint(0, N - 1)
            if new_c not in child:
                child.append(new_c)
        
        # Repair if invalid
        if not is_valid(child):
            child = repair(child)
            
        # Local Search Reduce
        child = local_search_reduce(child)
        
        child_len = len(child)
        
        # Acceptance Criterion
        # If better, always accept
        # If worse, accept with probability exp(-delta / T)
        if child_len < best_len:
            best_code = child[:]
            best_len = child_len
            current_code = child[:]
            # Reset temperature? Or just keep cooling? 
            # Resetting helps escape deep local minima.
            T = 5.0 
        elif child_len == best_len:
            # If same size, we might still accept to change the configuration
            # This helps in finding different local optima of the same size
            if random.random() < 0.1:
                current_code = child[:]
        else:
            # Worse
            delta = child_len - best_len
            # Probability of acceptance
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_code = child[:]
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Convert best_code indices back to lists of digits
    if best_code is None:
        # Fallback: return a trivial covering (all words) - though this is huge
        return [idx_to_digits(i) for i in range(N)]
        
    result = []
    for c in best_code:
        result.append(idx_to_digits(c))
        
    return result
