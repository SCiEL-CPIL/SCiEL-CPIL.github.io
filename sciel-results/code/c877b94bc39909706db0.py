import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for radius=1
    # neighbors[idx] = list of indices within Hamming distance <= 1
    space_size = q ** n
    
    # Powers of q for fast digit access
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    neighbors = np.empty(space_size, dtype=object)
    
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)
    
    # 1. Generate an initial valid code using greedy set cover
    # We use a boolean mask for uncovered to speed up greedy selection
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    
    # Precompute cover matrix for fast greedy updates
    # cover_masks[i][j] = True if j is in neighbors[i]
    cover_masks = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cover_masks[i, neighbors[i]] = True
        
    # Greedy Set Cover: pick word covering most uncovered
    while uncovered.any():
        counts = cover_masks @ uncovered.astype(np.int8)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        initial_code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - t_start > deadline - 5:
            break

    # Ensure we have a valid code
    if not uncovered.all():
        while not uncovered.all():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            w_idx = uncovered_indices[0]
            initial_code.append(w_idx)
            uncovered[neighbors[w_idx]] = False
            if time.time() - t_start > deadline - 5:
                break

    # 2. Fixed-Size ILS with Swap and Re-insertion
    # State: 
    # - code: list of codeword indices (fixed size K)
    # - cov: array of length space_size, cov[i] = number of codewords covering i
    # - in_set: boolean array, in_set[i] = True if i is in current_code
    
    def try_remove(code, cov, in_set_mask, c_idx):
        """Try to remove codeword c_idx. Returns (success, new_code, new_cov, new_in_set)."""
        c = code[c_idx]
        # Check if c is redundant: all its neighbors have cov > 1
        # If any neighbor has cov == 1, removing c makes it uncovered (cov=0)
        # So we need min(cov[neighbors[c]]) > 1
        if np.min(cov[neighbors[c]]) > 1:
            # Safe to remove
            new_code = code[:]
            new_code.pop(c_idx)
            
            new_cov = cov.copy()
            new_cov[neighbors[c]] -= 1
            
            new_in_set = in_set_mask.copy()
            new_in_set[c] = False
            
            return True, new_code, new_cov, new_in_set
        return False, None, None, None

    def try_swap(code, cov, in_set_mask, c_idx, new_c):
        """Try to swap codeword at c_idx with new_c. Returns (success, new_code, new_cov, new_in_set)."""
        if in_set_mask[new_c]:
            return False, None, None, None
            
        c = code[c_idx]
        if c == new_c:
            return False, None, None, None
            
        # Simulate removal of c and addition of new_c
        # We need to ensure that after removal, all neighbors of c are still covered
        # OR they become covered by new_c? No, swap is atomic.
        # Condition: For all x in neighbors[c], if cov[x] == 1, then x MUST be in neighbors[new_c].
        # Basically, the set of neighbors covered ONLY by c must be a subset of neighbors[new_c].
        
        # Let's check:
        # critical_nbs_c = [x for x in neighbors[c] if cov[x] == 1]
        # If any x in critical_nbs_c is NOT in neighbors[new_c], swap is invalid.
        
        # Optimization: 
        # We can compute the "exclusive" coverage of c.
        # Actually, just check:
        # For each x in neighbors[c]:
        #    if cov[x] == 1:
        #        if not cover_masks[new_c, x]: return False
        
        # This is O(deg(c) * check). deg is 15. Very fast.
        
        for x in neighbors[c]:
            if cov[x] == 1:
                if not cover_masks[new_c, x]:
                    return False, None, None, None
        
        # Swap is valid
        new_code = code[:]
        new_code[c_idx] = new_c
        
        new_cov = cov.copy()
        new_cov[neighbors[c]] -= 1
        new_cov[neighbors[new_c]] += 1
        
        new_in_set = in_set_mask.copy()
        new_in_set[c] = False
        new_in_set[new_c] = True
        
        return True, new_code, new_cov, new_in_set

    def try_insert(code, cov, in_set_mask, new_c):
        """Try to insert new_c into code. Returns (success, new_code, new_cov, new_in_set)."""
        if in_set_mask[new_c]:
            return False, None, None, None
            
        new_code = code + [new_c]
        new_cov = cov.copy()
        new_cov[neighbors[new_c]] += 1
        
        new_in_set = in_set_mask.copy()
        new_in_set[new_c] = True
        
        return True, new_code, new_cov, new_in_set

    # Initialize state
    current_code = initial_code[:]
    current_cov = np.zeros(space_size, dtype=np.int32)
    current_in_set = np.zeros(space_size, dtype=bool)
    for c in current_code:
        current_cov[neighbors[c]] += 1
        current_in_set[c] = True
        
    # Initial Pruning (Remove redundant words)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_code)):
            success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
            if success:
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                changed = True
                break # Restart loop after modification

    best_code = current_code[:]
    best_size = len(current_code)
    
    rng = np.random.default_rng(42)
    
    # ILS Loop
    # Moves:
    # 1. Try to remove a random word (if possible, size decreases)
    # 2. Try to swap a random word with a random other word (size same, might enable future removals)
    # 3. Try to insert a random word (size increases, might enable better swaps/removals later)
    # We accept moves if they improve (smaller size) or with some probability if neutral (simulated annealing style)
    # Given the strict time limit, simple hill climbing with random restarts/kicks is safer.
    
    # Strategy: 
    # - Keep current state.
    # - Generate a neighbor state via one move.
    # - If neighbor is valid (always is by construction of moves) and smaller, accept.
    # - If same size, accept with small probability (to wander).
    # - If larger, reject (unless we want to explore, but usually we want to minimize).
    # - After a series of no-improvements, do a "kick": randomize some words.
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - t_start < deadline - 5:
        # 1. Try to remove a random word
        if len(current_code) > 0:
            i = rng.integers(0, len(current_code))
            success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
            if success:
                # Improvement!
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                if len(current_code) < best_size:
                    best_code = current_code[:]
                    best_size = len(current_code)
                no_improve_count = 0
                continue
        
        # 2. Try to swap a random word with a random candidate
        if len(current_code) > 0:
            i = rng.integers(0, len(current_code))
            # Pick a random candidate from the whole space? Or from a local neighborhood?
            # Picking from whole space is more exploratory.
            new_c = rng.integers(0, space_size)
            success, new_code, new_cov, new_in_set = try_swap(current_code, current_cov, current_in_set, i, new_c)
            if success:
                # Same size. Accept with some probability to escape local minima?
                # Or just accept if it's "better" in some heuristic? 
                # For now, accept with prob 0.1 to allow wandering.
                if rng.random() < 0.1:
                    current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                # Check if this swap enables a removal?
                # We can try to remove any word in the new state.
                # But that's expensive. Let's just let the next iteration try removals.
                no_improve_count += 1
                continue
        
        # 3. Try to insert a random word (only if we are stuck and want to explore larger space?)
        # Actually, inserting increases size. We only want to do this if we think it will lead to a better configuration that can be pruned.
        # Let's do this rarely, e.g., when no_improve_count is high.
        if no_improve_count > max_no_improve:
            new_c = rng.integers(0, space_size)
            success, new_code, new_cov, new_in_set = try_insert(current_code, current_cov, current_in_set, new_c)
            if success:
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                no_improve_count = 0
                continue
        
        # If no move was made or accepted, increment no_improve
        no_improve_count += 1
        
        # If stuck for too long, do a random restart of the local search part?
        # Or just continue.
        
        # Additional check: if we have time, do a more aggressive local search:
        # Try to remove ANY word, not just random.
        # But we already do that in the "try_remove" loop at the start? 
        # No, we only do it once at the beginning. 
        # Let's add a "prune" step every N iterations.
        if no_improve_count % 10 == 0:
            changed = True
            while changed:
                changed = False
                for i in range(len(current_code)):
                    success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
                    if success:
                        current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                        changed = True
                        break
                if changed:
                    if len(current_code) < best_size:
                        best_code = current_code[:]
                        best_size = len(current_code)
                    no_improve_count = 0

    # Final check and return
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # Fallback to initial_code
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]
