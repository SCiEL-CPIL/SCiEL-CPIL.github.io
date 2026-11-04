import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Parameters
    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n-1-pos]) % q
        return w

    # Precompute neighbors for radius=1
    # For each index, store the list of indices within Hamming distance <= radius
    # Since radius=1, neighbors are self + 1-change neighbors
    # Optimization: We don't need to store all neighbors explicitly as arrays if we can compute marginal coverage efficiently.
    # However, for Greedy Set Cover, we need to know how many *new* points a candidate covers.
    # Let's precompute the neighbor lists as arrays of int32 for fast numpy operations.
    
    # Memory check: space_size * (1 + n*(q-1)) * 4 bytes
    # For q=3, n=7: 2187 * (1 + 7*2) * 4 = 2187 * 15 * 4 ~ 131 KB. Very small.
    
    neighbors = np.empty((space_size, 1 + n*(q-1)), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                # Calculate delta
                delta = (new_d - w[pos]) * q_powers[n-1-pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list
    
    # --- Phase 1: Greedy Set Cover Construction ---
    # Start with empty coverage
    uncovered = np.ones(space_size, dtype=bool)
    total_uncovered = space_size
    code_indices = []
    
    # We need to efficiently find the word that covers the most uncovered points.
    # Naive approach: For each candidate word, count overlap with 'uncovered'.
    # Since space is small (2187), we can do a full scan in each iteration.
    # Number of iterations will be roughly the code size (~150-200).
    # 200 * 2187 * 15 (avg neighbors) ~ 6.5M operations. Fast enough in Python/Numpy?
    # Let's try to vectorize the scoring.
    
    # Actually, iterating through all space_size candidates in pure Python for each step might be slow if code size is large.
    # But 2187 candidates is small.
    
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 10.0:
        best_idx = -1
        best_gain = 0
        
        # We can optimize by only considering candidates that are themselves uncovered or near uncovered?
        # No, a covered point can still be a good center if it covers many other uncovered points.
        
        # Vectorized gain calculation:
        # For each candidate c in 0..space_size-1, gain = sum(uncovered[neighbors[c]])
        # neighbors is (space_size, max_nb). 
        # We can compute this as:
        # gains = np.sum(uncovered[neighbors], axis=1)
        # But 'uncovered' is a boolean array of length space_size.
        # neighbors[c] gives indices.
        
        # Let's try this vectorization.
        # Note: neighbors array has fixed width. Some entries might be invalid if we didn't fill them? 
        # In the precomputation above, we filled exactly 1 + n*(q-1) entries for every idx. So it's a rectangular array.
        
        gains = np.sum(uncovered[neighbors], axis=1)
        
        # Find the index with max gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if total_uncovered > 0, but safety check
            break
            
        # Add this word to code
        code_indices.append(best_idx)
        
        # Update uncovered
        uncovered[neighbors[best_idx]] = False
        total_uncovered -= best_gain
        
    # Convert to list of words
    current_code = [idx_to_word(i) for i in code_indices]
    
    # --- Phase 2: Greedy Removal (Redundancy Check) ---
    # We might have added redundant words. Let's remove any word that is not the *only* cover for any point.
    # This is the standard "delete if min_coverage >= 2" pass.
    
    # Build coverage count array
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1
        
    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 5.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                # Safe to remove
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True
    
    current_code = [idx_to_word(code_indices[i]) for i in range(len(code_indices)) if active[i]]
    current_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    
    # --- Phase 3: Simulated Annealing Refinement ---
    # Try to improve by swapping codewords or moving them.
    # State: list of codeword indices.
    # Move: Pick a random codeword, replace it with a random other word (or move it to a random position).
    # Cost: Number of uncovered points. We want to minimize this (ideally 0).
    # Since we start from a valid code (0 uncovered), we only accept moves that keep uncovered=0 or use SA to explore.
    # Actually, if we start valid, we just want to find a *smaller* valid code.
    # So the objective is: Minimize Size, subject to Validity.
    # This is a constrained optimization.
    
    # Alternative SA approach: 
    # Objective: Penalty = Size + Lambda * (Uncovered Count)
    # Start with valid code. Try to remove a word. If it becomes invalid, calculate penalty.
    # Or: Start with valid code. Try swapping a word with a "better" word that allows removing another word?
    
    # Simpler SA: 
    # We want to find a subset of the current code (or a new set) that is smaller and valid.
    # Let's try a "random restart local search" on the set of codewords.
    # Move: Remove a random codeword. If valid, accept. If invalid, try to repair by adding a word that covers the newly uncovered points?
    # This is complex.
    
    # Let's stick to a simpler heuristic improvement:
    # 1. Randomly shuffle the code.
    # 2. Try to remove each word. If removal keeps code valid, remove it.
    # 3. If stuck, try to replace a word with a random word that increases coverage efficiency?
    
    # Given the time budget, let's do a few passes of "randomized greedy removal".
    # The deterministic removal in Phase 2 might get stuck in a local optimum depending on order.
    
    best_code = current_code[:]
    best_size = len(best_code)
    
    # Rebuild coverage for best_code
    cov_best = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_best[neighbors[c_idx]] += 1
        
    # Randomized removal attempts
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() - t_start > deadline - 3.0:
            break
            
        # Start from current best
        cur_code = best_code[:]
        cur_cov = cov_best.copy()
        
        # Shuffle order of removal attempts
        indices = list(range(len(cur_code)))
        np.random.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() - t_start > deadline - 3.0:
                break
            c = cur_code[i]
            c_idx = word_to_idx(c)
            nbs = neighbors[c_idx]
            
            if np.min(cur_cov[nbs]) >= 2:
                # Remove it
                cur_cov[nbs] -= 1
                # Remove from list (mark as removed to avoid shifting issues, or just filter at end)
                cur_code[i] = None 
                improved = True
        
        # Filter out None
        new_code = [c for c in cur_code if c is not None]
        
        if len(new_code) < best_size:
            best_code = new_code
            best_size = len(best_code)
            # Update cov_best
            cov_best = np.zeros(space_size, dtype=np.int32)
            for c in best_code:
                c_idx = word_to_idx(c)
                cov_best[neighbors[c_idx]] += 1
        
    # Final Verification
    if best_size == 0:
        # Fallback to initial greedy result if something went wrong
        return [idx_to_word(i) for i in code_indices]
        
    cov_final = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_final[neighbors[c_idx]] += 1
        
    if np.any(cov_final == 0):
        # Invalid, return the initial greedy code (which was valid)
        # Re-verify initial greedy code just in case
        cov_init = np.zeros(space_size, dtype=np.int32)
        for i in code_indices:
            cov_init[neighbors[i]] += 1
        if not np.any(cov_init == 0):
            return [idx_to_word(i) for i in code_indices]
        # If even that fails (shouldn't happen), return a trivial valid code (all words)
        return [idx_to_word(i) for i in range(space_size)]
        
    return best_code
