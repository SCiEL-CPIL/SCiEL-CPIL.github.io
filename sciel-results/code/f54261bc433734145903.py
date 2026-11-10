import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    total_words = q ** n
    # For q=3, n=7, total_words = 2187. This is small enough to precompute.
    
    # 1. Precompute representation
    # We represent words as integers 0..total_words-1 in base-q.
    # Precompute the "ball" for each word: a boolean vector of length total_words
    # indicating which words are within Hamming distance <= radius.
    
    # Generate all words as a matrix of digits (total_words, n)
    # This is efficient for small total_words.
    digits = np.array([[(w // (q ** i)) % q for i in range(n)] for w in range(total_words)], dtype=np.int8)
    
    # Precompute balls: balls[i] is a boolean array where balls[i][j] is True if dist(i, j) <= radius
    # dist(i, j) = sum(digits[i] != digits[j])
    # Memory: 2187 * 2187 booleans ~ 4.8 MB. Fine.
    
    # Compute pairwise differences
    # diff[i, j, k] = digits[i, k] != digits[j, k]
    # This is 2187 * 2187 * 7 ~ 33M elements. 
    # We can do this in chunks to save memory if needed, but 33M booleans is ~33MB, which is okay.
    
    # Vectorized distance calculation
    # To avoid huge temporary arrays, we can loop over dimensions or use broadcasting carefully.
    # Broadcasting: digits[:, None, :] vs digits[None, :, :]
    # Shape: (2187, 1, 7) vs (1, 2187, 7) -> (2187, 2187, 7)
    # This creates a 33MB boolean array. Acceptable.
    
    diff = digits[:, None, :] != digits[None, :, :]
    dist_matrix = np.sum(diff, axis=2).astype(np.int8)
    balls = dist_matrix <= radius
    
    # 2. Initial Solution
    # Start with a random or greedy solution.
    # Greedy is fast and provides a good starting point.
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    # Simple greedy initialization
    # To make it fast, we pick the word with max coverage, add it, update covered.
    # We can reuse the balls matrix.
    
    # Optimization: Instead of recalculating gain for all 2187 words every step (which is O(N^2)),
    # we can just do a few rounds of greedy or start with a random set and let SA do the work.
    # Given the short time limit, a simple random start or a very fast greedy is better.
    # Let's do a fast greedy:
    # We want to minimize the number of codewords.
    
    # Actually, for SA, a random start is often better to avoid bias, but might take longer to converge.
    # A hybrid: Start with a small random set, then let SA add/remove.
    # Or start with a greedy set.
    
    # Let's use a simple greedy to get a valid cover first, then optimize size.
    # This ensures we start with Energy = 0 (if coverage is perfect) or low.
    
    # Greedy Set Cover
    while not covered.all() and time.time() < deadline - 60:
        # Find word with max new coverage
        # gain[i] = sum(balls[i] & ~covered)
        # We can compute this vectorized:
        # (balls & ~covered[None, :]).sum(axis=1)
        # This is O(N^2) per step. ~2187*2187 = 4.8M ops. Fast in numpy.
        
        gain = (balls & ~covered[None, :]).sum(axis=1)
        if gain.max() == 0:
            break
        best_idx = np.argmax(gain)
        code_indices.append(best_idx)
        covered[balls[best_idx]] = True
        
    # If we ran out of time for greedy, we might have an incomplete cover.
    # SA will handle it.
    
    # Convert to binary vector
    x = np.zeros(total_words, dtype=bool)
    for idx in code_indices:
        x[idx] = True
        
    # 3. Simulated Annealing
    # State: x (binary vector)
    # Energy: 
    #   n_codewords = sum(x)
    #   n_uncovered = sum(cover_count == 0)
    #   E = n_codewords + 1000 * n_uncovered
    # We want to minimize E.
    
    # Maintain cover_count to efficiently update energy.
    cover_count = np.zeros(total_words, dtype=np.int16)
    for idx in code_indices:
        cover_count[balls[idx]] += 1
        
    current_energy = sum(x) + 1000 * np.sum(cover_count == 0)
    
    best_x = x.copy()
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    rng = np.random.default_rng(42)
    
    # Precompute indices of 1s and 0s for faster random selection?
    # Or just use rng.integers and check x[i].
    
    while time.time() < deadline - 2:
        if T < T_min:
            break
            
        # Move: Flip a random bit
        i = rng.integers(0, total_words)
        
        # Calculate delta energy
        # If x[i] is True (1), we are removing codeword i.
        #   n_codewords decreases by 1.
        #   cover_count[balls[i]] decreases by 1.
        #   Some words might become uncovered (if cover_count was 1 and becomes 0).
        #   Let new_uncovered = sum(cover_count[balls[i]] == 1)
        #   delta_uncovered = new_uncovered
        #   delta_E = -1 + 1000 * delta_uncovered
        
        # If x[i] is False (0), we are adding codeword i.
        #   n_codewords increases by 1.
        #   cover_count[balls[i]] increases by 1.
        #   Some words might become covered (if cover_count was 0 and becomes 1).
        #   Let old_uncovered_in_ball = sum(cover_count[balls[i]] == 0)
        #   delta_uncovered = -old_uncovered_in_ball
        #   delta_E = 1 + 1000 * delta_uncovered
        
        if x[i]:
            # Remove
            ball = balls[i]
            # Words that become uncovered are those with count 1 in this ball
            delta_uncovered = np.sum(cover_count[ball] == 1)
            delta_E = -1 + 1000 * delta_uncovered
            
            # Apply move
            x[i] = False
            cover_count[ball] -= 1
        else:
            # Add
            ball = balls[i]
            # Words that become covered are those with count 0 in this ball
            delta_uncovered = -np.sum(cover_count[ball] == 0)
            delta_E = 1 + 1000 * delta_uncovered
            
            # Apply move
            x[i] = True
            cover_count[ball] += 1
            
        new_energy = current_energy + delta_E
        
        # Acceptance
        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = x.copy()
        else:
            # Rollback
            if x[i]:
                # We added, now remove
                x[i] = False
                cover_count[balls[i]] -= 1
            else:
                # We removed, now add
                x[i] = True
                cover_count[balls[i]] += 1
                
        T *= cooling_rate
        
    # Extract best solution
    best_indices = np.where(best_x)[0]
    # Convert to list of lists
    result = []
    for idx in best_indices:
        # Convert int idx to base-q digits
        w = []
        for i in range(n):
            w.append((idx // (q ** i)) % q)
        result.append(w)
        
    return result
