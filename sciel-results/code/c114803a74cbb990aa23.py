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

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    # Precompute neighbors for radius=1
    # neighbors[idx] = list of indices within Hamming distance <= 1
    space_size = q ** n
    neighbors = np.empty(space_size, dtype=object)
    
    # Powers of q for fast digit access
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

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
    # Since space_size is 2187, we can use a dense boolean array
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    
    # Precompute a list of neighbor arrays for quick access
    # Greedy Set Cover: pick word covering most uncovered
    while uncovered.any():
        # Find word covering most uncovered words
        # counts[i] = number of uncovered words covered by i
        # We can compute this by summing uncovered over neighbors of i
        # But doing this for all i every step is O(N^2 * deg)
        # N=2187, deg=1+7*2=15. 2187*15*2187 ~ 70M ops. 
        # With numpy: 
        # cover_matrix is (N, N) boolean. 
        # counts = cover_matrix @ uncovered.astype(int)
        
        # To save memory/time, we can just iterate and compute on the fly if N is small
        # But N=2187 is small enough for a dense matrix if we optimize.
        
        # Let's use the dense matrix approach for speed in the greedy phase
        # It was used in the previous version and worked.
        
        # Actually, let's just do a simple loop to find the best candidate
        # since N is small, and we only do this until covered.
        
        best_idx = -1
        max_count = 0
        
        # Optimization: Only consider indices that are not yet in code? 
        # No, a code word can cover words even if it's not itself uncovered.
        # But adding a word already in code is useless.
        
        # Let's precompute the cover matrix once to make greedy fast.
        # cover_masks[i][j] = True if j is in neighbors[i]
        # This is 2187x2187 boolean = ~4.7MB. Acceptable.
        
        if 'cover_masks' not in locals():
             cover_masks = np.zeros((space_size, space_size), dtype=bool)
             for i in range(space_size):
                 cover_masks[i, neighbors[i]] = True
        
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

    # 2. Iterated Local Search for minimization
    # Representation:
    # - current_code: list of codeword indices
    # - current_cov: array of length space_size, current_cov[i] = number of codewords covering i
    # - in_set: boolean array, in_set[i] = True if i is in current_code
    
    current_code = initial_code[:]
    current_cov = np.zeros(space_size, dtype=np.int32)
    in_set = np.zeros(space_size, dtype=bool)
    
    for c in current_code:
        current_cov[neighbors[c]] += 1
        in_set[c] = True
    
    best_code = current_code[:]
    best_size = len(current_code)
    
    rng = np.random.default_rng(42)
    
    # Local Search: Pruning
    # A codeword c is "redundant" (removable) if for all neighbors x of c, current_cov[x] > 1.
    # If we remove c, current_cov[x] becomes current_cov[x] - 1.
    # So we need min(current_cov[neighbors[c]]) > 1.
    
    def local_search(code, cov, in_set_mask):
        # code: list of indices
        # cov: array of coverage counts
        # in_set_mask: boolean array
        # Returns: pruned_code, pruned_cov, pruned_in_set
        
        # We repeatedly remove redundant words until none exist.
        # To make this efficient, we can maintain a set of "critical" words.
        # A word x is critical if cov[x] == 1.
        # A codeword c is removable if it covers no critical words.
        
        # Let's maintain a list of critical words (where cov == 1)
        # Actually, it's easier to just check removability directly.
        
        # Optimization: 
        # If we remove c, cov[neighbors[c]] decreases by 1.
        # New critical words are those in neighbors[c] where cov was 2.
        
        # Let's use a queue of candidate words to check for removal.
        # Initially, all words in code are candidates.
        
        # We'll use a simple loop:
        # 1. Find a removable word.
        # 2. Remove it.
        # 3. Repeat.
        
        # To find a removable word fast:
        # A word c is NOT removable if it is the ONLY cover for some neighbor x.
        # i.e., if there exists x in neighbors[c] such that cov[x] == 1.
        
        # Let's maintain a count of "critical neighbors" for each codeword?
        # critical_count[c] = number of neighbors x of c such that cov[x] == 1.
        # If critical_count[c] == 0, c is removable.
        
        # Initialize critical_count
        critical_count = np.zeros(space_size, dtype=np.int32)
        
        # Find all critical words (cov == 1)
        critical_words = np.where(cov == 1)[0]
        
        # For each critical word x, increment critical_count for all codewords covering x.
        # Who covers x? The codewords c such that x in neighbors[c].
        # This is the reverse neighbor map.
        # reverse_neighbors[x] = list of codewords c covering x.
        
        # Precompute reverse neighbors?
        # reverse_neighbors[x] is just neighbors[x] restricted to in_set?
        # No, neighbors is symmetric. x in neighbors[c] iff c in neighbors[x].
        # So the codewords covering x are in_set[neighbors[x]].
        
        for x in critical_words:
            nbs = neighbors[x]
            # Only count those that are in the set
            for c in nbs:
                if in_set_mask[c]:
                    critical_count[c] += 1
        
        # Now, find a c in code with critical_count[c] == 0
        # We can just iterate through code list
        
        changed = True
        while changed:
            changed = False
            # Find first removable
            for i, c in enumerate(code):
                if critical_count[c] == 0:
                    # Remove c
                    code.pop(i)
                    in_set_mask[c] = False
                    cov[neighbors[c]] -= 1
                    
                    # Update critical_count for affected neighbors
                    # The neighbors of c might have changed coverage.
                    # Specifically, neighbors where cov became 1 (new critical) or was 1 (no longer critical).
                    
                    # For each neighbor x of c:
                    # Old cov: cov[x] + 1 (since we just subtracted 1, current is new)
                    # Wait, cov is already updated.
                    # Let's track changes.
                    
                    # Before removal: cov_old[x] = cov_new[x] + 1
                    # If cov_new[x] == 1, then cov_old[x] == 2. x is NEWLY critical.
                    # If cov_new[x] == 0, then cov_old[x] == 1. x is NO LONGER critical.
                    # (cov_new[x] cannot be < 0 if valid)
                    
                    nbs_c = neighbors[c]
                    for x in nbs_c:
                        if cov[x] == 1:
                            # x is newly critical
                            # Who covers x? All in_set[neighbors[x]]
                            # We need to increment critical_count for all codewords covering x
                            # EXCEPT c (which is removed)
                            # But c is no longer in_set, so in_set[neighbors[x]] excludes c.
                            # So we increment for all in_set[neighbors[x]]
                            for c2 in neighbors[x]:
                                if in_set_mask[c2]:
                                    critical_count[c2] += 1
                        elif cov[x] == 0:
                            # x was critical (cov_old=1), now cov_new=0?
                            # This would mean the code became invalid!
                            # But we only removed if critical_count[c]==0, meaning c covered NO critical words.
                            # So cov_old[x] could not have been 1 for any x in neighbors[c].
                            # Thus cov_new[x] cannot be 0.
                            pass
                    
                    changed = True
                    break # Restart loop to find next removable
            
        return code, cov, in_set_mask

    # Kick: Randomly add/remove some words to escape local minima
    def kick(code, cov, in_set_mask, strength):
        # strength: number of random moves
        # We perform 'strength' random add/remove pairs or single moves
        # To maintain validity, we can:
        # 1. Remove a random word (might make invalid)
        # 2. Add a random word (always valid if it covers uncovered)
        
        # Better kick for strictly valid ILS:
        # 1. Identify a set of "weak" words (those covering many critical words? or just random).
        # 2. Remove K random words.
        # 3. Add K random words that cover the newly uncovered spots.
        
        # Simpler kick:
        # Pick K random words from current code, remove them.
        # Then greedily cover any uncovered words.
        
        K = strength
        if len(code) < K:
            return code, cov, in_set_mask
        
        # Pick K indices to remove
        # We should pick ones that are "least critical" to minimize repair cost?
        # Or just random.
        
        # Let's try removing words that have high critical_count? No, that breaks it more.
        # Remove words with low critical_count?
        
        # Let's just remove K random words.
        # Then repair.
        
        # Save state
        old_code = code[:]
        old_cov = cov.copy()
        old_in_set = in_set_mask.copy()
        
        # Remove K random words
        indices_to_remove = rng.choice(len(code), size=min(K, len(code)), replace=False)
        # Sort descending to remove from end first to avoid index shifting issues
        indices_to_remove = np.sort(indices_to_remove)[::-1]
        
        for idx in indices_to_remove:
            c = code[idx]
            code.pop(idx)
            in_set_mask[c] = False
            cov[neighbors[c]] -= 1
        
        # Repair: Cover uncovered words
        # uncovered = np.where(cov == 0)[0]
        
        # Greedy repair
        while True:
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a word that covers the most uncovered
            # counts = cover_masks @ (cov == 0).astype(int)
            # But we don't have cover_masks in scope here? Yes we do.
            
            # Optimization: Only look at words not in set?
            # Actually, adding a word already in set is useless.
            
            # Let's just use the precomputed cover_masks
            # But cover_masks is global.
            
            # To speed up, we can restrict candidates to neighbors of uncovered words?
            # Any word covering an uncovered word must be a neighbor of it.
            
            # Let's just use the dense matrix multiplication. It's fast for N=2187.
            uncovered_bool = (cov == 0)
            counts = cover_masks @ uncovered_bool.astype(np.int8)
            
            # Mask out words already in set?
            # If a word is in set, it's already covering things. Adding it again does nothing.
            # But in_set is boolean.
            # counts[in_set] = 0 ?
            # No, if in_set[c] is True, we shouldn't add it.
            
            # Let's just pick the best. If it's in set, we have a problem.
            # But if it's in set, it's already covering. Why is there uncovered?
            # Because other words were removed.
            # So the best word to add should NOT be in set.
            
            # Set counts for in_set words to -1 or 0?
            # If we set to 0, and there are uncovered words, the max will be > 0 for some non-set word.
            # Unless the only coverers are in set? Impossible, if in set, they cover it.
            
            counts[in_set_mask] = -1
            
            best_idx = np.argmax(counts)
            
            if counts[best_idx] <= 0:
                # Should not happen if valid
                break
            
            code.append(best_idx)
            in_set_mask[best_idx] = True
            cov[neighbors[best_idx]] += 1
            
            if time.time() - t_start > deadline - 2:
                break

        return code, cov, in_set_mask

    # Main ILS Loop
    current_code, current_cov, current_in_set = local_search(current_code, current_cov, in_set)
    best_code = current_code[:]
    best_size = len(current_code)
    
    kick_strength = 3
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - t_start < deadline - 5:
        # Kick
        kicked_code, kicked_cov, kicked_in_set = kick(current_code, current_cov, current_in_set, kick_strength)
        
        # Local Search
        pruned_code, pruned_cov, pruned_in_set = local_search(kicked_code, kicked_cov, kicked_in_set)
        
        # Check improvement
        if len(pruned_code) < len(best_code):
            best_code = pruned_code[:]
            best_size = len(pruned_code)
            no_improve_count = 0
            # Accept the kick
            current_code, current_cov, current_in_set = pruned_code, pruned_cov, pruned_in_set
        else:
            no_improve_count += 1
            # If we've had many no-improvements, increase kick strength
            if no_improve_count > max_no_improve:
                kick_strength += 1
                if kick_strength > 10:
                    kick_strength = 3
                no_improve_count = 0
        
        # If time is running out, break
        if time.time() - t_start > deadline - 10:
            break

    # Final check and return
    # Ensure best_code is valid
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # Fallback to initial_code
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]
