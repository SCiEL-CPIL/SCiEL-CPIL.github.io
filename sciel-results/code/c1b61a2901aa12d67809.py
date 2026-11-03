import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Total words
    total_words = q ** n
    
    # Precompute neighbors using numpy
    # All words as indices
    all_indices = np.arange(total_words, dtype=np.int64)
    
    # Compute neighbors for each position
    # For position p, the "stride" is q^p
    # Changing digit at position p: new_idx = idx - digit*q^p + new_digit*q^p
    # We can compute this for all positions at once
    
    # Precompute: for each position p, and each word, the neighbor indices
    # But we need the ball mask (OR of all neighbors)
    
    # Better: compute the set of neighbors for each word
    # Ball size: 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    # For each word, compute its neighbors
    # We'll store ball_masks as a list of bitmasks (Python ints)
    # But for numpy speed, we'll compute coverage differently
    
    # Strategy: 
    # 1. Precompute for each word its neighbors as a flat array
    # 2. For greedy, maintain covered as a boolean numpy array
    # 3. For each candidate, compute gain = number of uncovered neighbors
    
    # Precompute neighbor lists
    # neighbors[i] is a list of neighbor indices for word i
    # But for speed, we'll use a 2D array: neighbor_matrix[i, j] = j-th neighbor of i
    
    # First, generate all words
    digits = np.array(list(range(q)))
    
    # Build neighbor matrix
    # For each position p, for each word, the neighbors by changing that digit
    # total neighbors = 1 + n*(q-1)
    
    # Precompute powers
    powers = np.array([q ** p for p in range(n)], dtype=np.int64)
    
    # For each word index, extract digits
    # word_digits[i, p] = digit at position p of word i
    word_digits = np.zeros((total_words, n), dtype=np.int64)
    for p in range(n):
        word_digits[:, p] = (all_indices // powers[p]) % q
    
    # Build neighbor indices
    # For each position p, for each word, for each new_digit != current_digit
    # neighbor_idx = word_idx - current_digit * powers[p] + new_digit * powers[p]
    
    # We'll build a flat array of all neighbors
    # neighbors_flat[i * ball_size + j] = j-th neighbor of word i
    
    # Start with the word itself
    neighbors_flat = np.zeros((total_words, ball_size), dtype=np.int64)
    neighbors_flat[:, 0] = all_indices
    
    offset = 1
    for p in range(n):
        current_digits = word_digits[:, p]
        for new_digit in range(q):
            # Words where current digit != new_digit
            mask = current_digits != new_digit
            if not np.any(mask):
                continue
            # For those words, the neighbor index
            new_indices = all_indices[mask] - current_digits[mask] * powers[p] + new_digit * powers[p]
            neighbors_flat[mask, offset] = new_indices
            offset += 1
    
    # Now neighbors_flat[i, :] contains all neighbors of word i
    # Note: some entries might be uninitialized if we didn't fill all, but we should have filled all
    
    # For the greedy algorithm, we need to compute gain quickly
    # gain[i] = number of j in neighbors_flat[i] such that covered[j] == False
    
    # We can use numpy: for each i, gain[i] = sum(~covered[neighbors_flat[i, :]])
    # This is a gather operation
    
    # covered will be a boolean numpy array of size total_words
    
    # Let's also precompute the ball_masks as Python ints for the local search
    # Actually, for local search we can also use numpy
    
    # Convert neighbors_flat to a list of lists for easier access in local search
    # Or keep as numpy and use fancy indexing
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    def greedy_cover(seed):
        random.seed(seed)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Compute gain for all candidates
            # gain[i] = number of uncovered neighbors of i
            # Using fancy indexing: ~covered[neighbors_flat[i, :]]
            # This creates a 2D boolean array, then sum along axis 1
            
            # For efficiency, we can do:
            # uncovered_neighbors = ~covered[neighbors_flat]  # shape (total_words, ball_size)
            # gains = uncovered_neighbors.sum(axis=1)
            
            # But this creates a large temporary array. Let's try it.
            uncovered_mask = ~covered[neighbors_flat]  # (total_words, ball_size)
            gains = uncovered_mask.sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Should not happen if not all covered
                break
            
            # Find all candidates with max gain
            best_candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best candidates
            chosen = int(random.choice(best_candidates))
            code.append(chosen)
            
            # Update covered
            covered[neighbors_flat[chosen, :]] = True
            
            if time.time() > deadline:
                return None
        
        return code
    
    def reduce_cover(code):
        if not code:
            return code
        
        # cover_count: for each word, how many codewords cover it
        cover_count = np.zeros(total_words, dtype=np.int64)
        
        for c in code:
            cover_count[neighbors_flat[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any word in ball(c) has cover_count == 1
                ball_words = neighbors_flat[c, :]
                if np.min(cover_count[ball_words]) > 1:
                    # Redundant
                    cover_count[ball_words] -= 1
                    code.pop(i)
                    changed = True
        
        return code
    
    def local_search(code):
        if not code:
            return code
        
        cover_count = np.zeros(total_words, dtype=np.int64)
        for c in code:
            cover_count[neighbors_flat[c, :]] += 1
        
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each codeword
            for i in range(len(code)):
                c = code[i]
                ball_words = neighbors_flat[c, :]
                # Words that would become uncovered
                loss_words = ball_words[cover_count[ball_words] == 1]
                
                if len(loss_words) == 0:
                    # Redundant
                    cover_count[ball_words] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break  # Restart loop
            
            if improved:
                continue
            
            # Try swaps
            for i in range(len(code)):
                c = code[i]
                ball_words = neighbors_flat[c, :]
                loss_words = ball_words[cover_count[ball_words] == 1]
                
                if len(loss_words) == 0:
                    continue  # Should have been caught above
                
                # Find a c' not in code that covers all loss_words
                # For each candidate c', check if loss_words is subset of neighbors_flat[c', :]
                
                # We can vectorize: for each candidate, compute how many loss_words it covers
                # But loss_words varies per i, so we do this per i
                
                # Create a boolean mask for loss_words
                loss_mask = np.zeros(total_words, dtype=bool)
                loss_mask[loss_words] = True
                
                # For candidates not in code, check if they cover all loss_words
                # candidates = ~in_code
                # For each candidate, count how many loss_words are in its ball
                # covered_loss = sum(loss_mask[neighbors_flat[candidate, :]])
                # We want covered_loss == len(loss_words)
                
                # Vectorized: 
                # candidate_indices = np.where(~in_code)[0]
                # For each candidate, neighbors_flat[candidate, :] gives its ball
                # We need: all loss_words are in the ball
                # Equivalent: sum(loss_mask[neighbors_flat[candidate, :]]) == len(loss_words)
                
                candidate_indices = np.where(~in_code)[0]
                if len(candidate_indices) == 0:
                    continue
                
                # For each candidate, compute coverage of loss_words
                # This is a gather: loss_mask[neighbors_flat[candidate_indices, :]]
                # Shape: (len(candidate_indices), ball_size)
                candidate_cover = loss_mask[neighbors_flat[candidate_indices, :]]
                coverage_counts = candidate_cover.sum(axis=1)
                
                # Find candidates that cover all loss_words
                good_candidates = np.where(coverage_counts == len(loss_words))[0]
                
                if len(good_candidates) == 0:
                    continue
                
                # Pick the first good candidate (or random)
                best_idx = int(good_candidates[0])
                best_c_prime = int(candidate_indices[best_idx])
                
                # Perform swap
                # Remove c
                cover_count[neighbors_flat[c, :]] -= 1
                in_code[c] = False
                code[i] = best_c_prime
                
                # Add c_prime
                cover_count[neighbors_flat[best_c_prime, :]] += 1
                in_code[best_c_prime] = True
                
                # Check if this swap enabled a reduction
                reduced = False
                for j in range(len(code)):
                    cj = code[j]
                    ball_words_j = neighbors_flat[cj, :]
                    if np.min(cover_count[ball_words_j]) > 1:
                        # Redundant
                        cover_count[ball_words_j] -= 1
                        in_code[cj] = False
                        code.pop(j)
                        reduced = True
                        break
                        
                if reduced:
                    improved = True
                    break  # Restart main while loop
                else:
                    # Revert swap
                    cover_count[neighbors_flat[best_c_prime, :]] -= 1
                    in_code[best_c_prime] = False
                    code[i] = c
                    cover_count[neighbors_flat[c, :]] += 1
                    in_code[c] = True
        
        return code
    
    # Main loop
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
        
    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
