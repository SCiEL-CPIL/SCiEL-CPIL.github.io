import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Specific to the task: q=3, n=8, radius=1
    # Total words = 3^8 = 6561
    total_words = q ** n

    # Precompute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbors (the ball of radius 1) for each index
    # neighbors[idx] is a numpy array of 17 indices
    ball_size = 1 + n * (q - 1)
    neighbors = np.empty((total_words, ball_size), dtype=np.int32)
    
    # Also precompute digits for conversion back to list at the end
    digits_of = [None] * total_words
    
    for idx in range(total_words):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        # Store reversed for potential use, but we need standard order for output
        # The task says "codeword is a list of n integers". Usually index 0 corresponds to most significant or least?
        # In the seed: 
        # d.append(x % q) ... d.reverse()
        # So digits_of[i] is [d_0, d_1, ..., d_7] where i = d_0*q^7 + ... + d_7?
        # Let's check:
        # x = i
        # d.append(x % q) -> d_0 (least significant in this loop)
        # x //= q
        # ...
        # d.reverse() -> [most significant, ..., least significant]
        # So if i = 0, d = [0,0,0,0,0,0,0,0]
        # If i = 1, d = [0,0,0,0,0,0,0,1]
        # This implies index i corresponds to word w where w[j] is the digit at position j (0-indexed from left).
        # i = sum(w[j] * q^(n-1-j))
        
        # Let's verify powers usage.
        # If we use powers[i] = q^i, then index = sum(digit[i] * powers[i]) means digit[0] is least significant.
        # But digits_of stores reversed, so digits_of[i][0] is most significant.
        # Let's stick to the seed's convention to be safe, or just ensure consistency.
        
        # In the seed:
        # idx = 0
        # for j in range(n): idx = idx * q + new_d[j]
        # This means new_d[0] is the most significant digit.
        # So index i = d_0 * q^7 + d_1 * q^6 + ... + d_7 * q^0.
        
        # Let's precompute neighbors using this index scheme.
        # To change digit at position pos (0 is most significant), the weight is q^(n-1-pos).
        
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            weight = q ** (n - 1 - pos)
            
            # Try all other values
            for new_val in range(q):
                if new_val == orig_digit:
                    continue
                delta = (new_val - orig_digit) * weight
                n_idx = idx + delta
                neighbors[idx, neighbor_idx] = n_idx
                neighbor_idx += 1
        
        # Store digits in the format expected by the output: list of ints
        # The seed returns [list(digits_of[i]) for i in final_code]
        # digits_of[i] was built as tuple(d) where d was reversed from least-significant-first.
        # So digits_of[i] = [d_0, d_1, ..., d_7] corresponding to most-significant-first.
        digits_of[idx] = tuple(word)

    # Build reverse mapping: for each point, which balls contain it?
    # This is expensive to store fully (6561 * 17 entries).
    # Instead, for swap operations, we need to find c2 such that c2 covers risky points.
    # Risky points are x. c2 must be in neighbors[x].
    # So for a set of risky points R, c2 must be in intersection_{x in R} neighbors[x].
    # We can compute this on the fly using the neighbors array.

    # --- Phase 1: Fast Greedy Initialization ---
    # We use a potential-based greedy.
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # potential[i] = number of neighbors of i that are currently uncovered
    potential = np.full(total_words, ball_size, dtype=np.int32)
    uncovered_remaining = total_words
    
    # To speed up the "find max potential" step, we could use a heap, but np.argmax is O(N)=6561 which is fast.
    # However, updating potential is the bottleneck.
    
    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        
        neigh_c = neighbors[c]
        # For each point x covered by c:
        # If x was uncovered (cov_count[x] == 0), it becomes covered.
        # This decreases the potential of all y that cover x (i.e., y in neighbors[x]).
        # We need to iterate through neighbors[x] for each x in neigh_c.
        # This is 17 * 17 = 289 operations per added codeword.
        
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Update potentials for all y in neighbors[x]
                for y in neighbors[x]:
                    if is_code[y]:
                        continue # Don't update potential of existing codewords, or does it matter?
                    # The potential of y is the count of its neighbors that are uncovered.
                    # x was uncovered, now covered. So potential[y] decreases by 1.
                    potential[y] -= 1
            else:
                cov_count[x] += 1
                
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # --- Phase 2: Iterated Local Search ---
    # We maintain cov_count to check if a codeword can be removed.
    # A codeword c can be removed if all x in neighbors[c] have cov_count[x] > 1.
    
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    # We will use a random search with removal and swap.
    # To make swaps effective, we need to find c2 that covers the "risky" points of c1.
    
    stuck_counter = 0
    
    # Precompute a list of indices for random access
    # We'll use random.choice on the list
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size > 0:
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]
            
            # Check if c is redundant
            # np.all(current_cov_count[neighbors[c]] > 1)
            # This is a vectorized check over 17 elements.
            if np.all(current_cov_count[neighbors[c]] > 1):
                # Remove c
                current_cov_count[neighbors[c]] -= 1
                current_is_code[c] = False
                current_code_set.remove(c)
                current_code.pop(idx)
                current_size -= 1
                stuck_counter = 0
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                stuck_counter += 1
                
        # 2. If stuck, try a swap
        # We need to find c1 in code and c2 not in code such that replacing c1 with c2 maintains coverage.
        # Condition: For all x in neighbors[c1], if cov_count[x] == 1 (only covered by c1), then x must be in neighbors[c2].
        # Equivalently: neighbors[c1] \ neighbors[c2] must be a subset of {x | cov_count[x] > 1}.
        
        if stuck_counter > 20 and time.time() < deadline - 10.0 and current_size > 0:
            # Pick a random c1 to try to replace
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Identify risky points: x in neighbors[c1] where cov_count[x] == 1
            # These points will lose coverage if c1 is removed.
            # We need c2 to cover all of them.
            neigh_c1 = neighbors[c1]
            risky_mask = current_cov_count[neigh_c1] == 1
            risky_points = neigh_c1[risky_mask]
            
            if len(risky_points) == 0:
                # c1 is redundant, but we didn't remove it? 
                # Maybe we just missed it or it became redundant after a swap?
                # Try to remove it again.
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find candidates c2 not in code that cover all risky points.
                # c2 must be in intersection of neighbors[x] for x in risky_points.
                # Start with neighbors of first risky point
                if len(risky_points) == 1:
                    candidates = set(neighbors[risky_points[0]].tolist())
                else:
                    candidates = set(neighbors[risky_points[0]].tolist())
                    for x in risky_points[1:]:
                        candidates &= set(neighbors[x].tolist())
                        if not candidates:
                            break
                
                if candidates:
                    # Filter out candidates already in code
                    # Also, c2 should ideally be different from c1 (though if c1 is in candidates, it's not a valid "new" codeword to add, we are replacing)
                    # Actually, if c1 is in candidates, it means c1 covers all risky points (which is true). But we can't add c1 again.
                    valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
                    
                    if valid_candidates:
                        # Pick a random valid candidate
                        c2 = random.choice(valid_candidates)
                        
                        # Perform the swap
                        # Remove c1
                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        
                        # Add c2
                        current_cov_count[neighbors[c2]] += 1
                        current_is_code[c2] = True
                        current_code_set.add(c2)
                        
                        # Update code list
                        current_code[c1_idx] = c2
                        
                        stuck_counter = 0
                    else:
                        # No valid swap found for this c1
                        pass
                else:
                    # No candidate covers all risky points
                    pass
            
            # Reset stuck counter to allow more removal attempts?
            # Or keep it high to try more swaps?
            # Let's reset to 0 to balance.
            stuck_counter = 0

    # Final Polish: Iterative removal
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Iterate backwards to safely remove
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
                
    # Verify coverage
    if not np.all(final_cov_count > 0):
        # Fallback to best_code
        # Re-verify best_code just in case
        bc_cov = np.zeros(total_words, dtype=np.int32)
        for idx in best_code:
            bc_cov[neighbors[idx]] += 1
        if not np.all(bc_cov > 0):
            # Should not happen, but return something valid if possible
            # Greedy full coverage
            is_covered = np.zeros(total_words, dtype=bool)
            full_code = []
            for b in range(total_words):
                if np.all(is_covered):
                    break
                if np.any(~is_covered[neighbors[b]]):
                    full_code.append(b)
                    is_covered[neighbors[b]] = True
            return [list(digits_of[i]) for i in full_code]
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
