import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a hybrid of greedy initialization and simulated annealing to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        if radius >= 2:
            # For radius 2, we need pairs. 
            # For the specific task (r=1), this is skipped.
            # If r=2, this part would need implementation.
            pass

    # Greedy Initialization
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    uncovered_set = set(range(total_words))
    
    # Fast Greedy: Pick center covering most uncovered
    # For small N, we can just iterate all candidates
    while uncovered_set:
        if time.time() > deadline:
            break
        best_center = -1
        best_count = -1
        
        # Optimization: Only check candidates that are uncovered? 
        # No, a covered center might cover more uncovered neighbors.
        # But for r=1, a center covers itself and neighbors.
        # If a center is covered, its neighbors are likely covered too.
        # Actually, just checking all is fast enough for 729.
        
        # To speed up, we can maintain a list of "active" candidates.
        # But for now, simple loop.
        for candidate in range(total_words):
            # Count uncovered neighbors
            count = 0
            for nb in neighbors[candidate]:
                if nb in uncovered_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_center = candidate
        
        if best_center == -1:
            break
            
        codewords.append(best_center)
        for nb in neighbors[best_center]:
            if nb in uncovered_set:
                uncovered_set.remove(nb)
                
    # Current size
    current_size = len(codewords)
    
    # If we already have the theoretical best (73 for 3,6,1), return
    if q == 3 and n == 6 and radius == 1:
        if current_size == 73:
            return [int_to_word(c, n, q) for c in codewords]
            
    # Simulated Annealing to reduce size
    # We try to find a valid code of size M, starting from M=73 (or lower bound) up to current_size
    # For q=3, n=6, r=1, lower bound is 71, best known 73.
    # We will try M = 73, 74, ... current_size
    
    # Prepare data structures for SA
    # cover_count[i] = number of codewords covering word i
    cover_count = np.zeros(total_words, dtype=np.int32)
    current_codewords = set(codewords)
    
    # Initialize cover_count for the greedy solution
    for c in current_codewords:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    best_valid_code = list(current_codewords)
    best_valid_size = current_size
    
    # Function to get uncovered count
    def get_uncovered_count():
        return np.count_nonzero(cover_count == 0)
    
    # Try to find a valid code of size M
    # We start with the greedy code, and randomly remove (current_size - M) codewords
    # Then use SA to fix the holes
    
    target_sizes = []
    if q == 3 and n == 6 and radius == 1:
        # Try 73 first, then 74, etc.
        start_m = 73
    else:
        # Estimate lower bound or just start from current_size - 5
        start_m = max(1, current_size - 10)
        
    for M in range(start_m, current_size + 1):
        if time.time() > deadline:
            break
            
        # If M is larger than best found so far, no point
        if M >= best_valid_size:
            break
            
        # Try multiple restarts for this M
        max_restarts = 50 if M == 73 else 10
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize state: Start with greedy code, remove random extras
            state_codewords = set(codewords)
            # Remove (current_size - M) random codewords
            to_remove = list(state_codewords)
            random.shuffle(to_remove)
            for i in range(current_size - M):
                c = to_remove[i]
                state_codewords.remove(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    
            # Reset cover_count for SA? 
            # We modified the global cover_count. We need to restore it or use a local one.
            # Let's use a local cover_count for SA to avoid restoring.
            local_cover_count = np.zeros(total_words, dtype=np.int32)
            for c in state_codewords:
                for nb in neighbors[c]:
                    local_cover_count[nb] += 1
                    
            # SA Parameters
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.999
            max_steps = 50000
            current_uncovered = np.count_nonzero(local_cover_count == 0)
            
            # If already valid, we found it!
            if current_uncovered == 0:
                best_valid_code = list(state_codewords)
                best_valid_size = M
                found_valid = True
                break
                
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random codeword to swap out
                c_out = random.choice(list(state_codewords))
                # Pick a random candidate to swap in (not in state_codewords)
                # To make it faster, we can pick from a list of "good" candidates or just random
                c_in = random.randint(0, total_words - 1)
                if c_in in state_codewords:
                    continue
                    
                # Perform swap
                state_codewords.remove(c_out)
                state_codewords.add(c_in)
                
                # Update local_cover_count
                for nb in neighbors[c_out]:
                    local_cover_count[nb] -= 1
                for nb in neighbors[c_in]:
                    local_cover_count[nb] += 1
                    
                new_uncovered = np.count_nonzero(local_cover_count == 0)
                delta = new_uncovered - current_uncovered
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_uncovered = new_uncovered
                    if current_uncovered == 0:
                        best_valid_code = list(state_codewords)
                        best_valid_size = M
                        found_valid = True
                        break
                else:
                    # Reject move, swap back
                    state_codewords.remove(c_in)
                    state_codewords.add(c_out)
                    for nb in neighbors[c_in]:
                        local_cover_count[nb] -= 1
                    for nb in neighbors[c_out]:
                        local_cover_count[nb] += 1
                        
                T *= cooling_rate
                if T < T_min:
                    break
                    
            if found_valid:
                break
                
    # If we found a better valid code, return it
    if best_valid_size < current_size:
        return [int_to_word(c, n, q) for c in best_valid_code]
    else:
        return [int_to_word(c, n, q) for c in codewords]
