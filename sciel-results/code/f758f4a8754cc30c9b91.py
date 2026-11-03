import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_size = 1 + n * (q - 1)
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors as a NumPy array for vectorized operations
    # neighbors_arr[i, j] is the index of the j-th word covered by codeword i
    neighbors_arr = np.zeros((total_words, cover_size), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Precompute reverse cover: for each word, which codewords cover it
    # This is useful for finding swaps that fix multiple uncovered words at once
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in range(cover_size):
            w = neighbors_arr[i, j]
            reverse_cover[w].append(i)
            
    # Convert reverse_cover to lists of numpy arrays for faster intersection if needed, 
    # but for now, standard lists are fine as the intersection is small.
    
    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation: how many currently uncovered words does each candidate cover?
            # neighbors_arr has shape (total_words, cover_size)
            # uncovered[neighbors_arr] has shape (total_words, cover_size)
            # sum(axis=1) gives the count of covered-but-still-uncovered words for each candidate
            gains = uncovered[neighbors_arr].sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            # Mark words covered by 'chosen' as no longer uncovered
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
        """Remove redundant codewords. A codeword is redundant if all words it covers are covered by others."""
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                # If any word has count 0, this codeword is essential
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """
        Iterative local search:
        1. Try to remove a codeword.
        2. If removal creates uncovered words, try to find a replacement codeword that covers all of them.
        3. If no replacement found, try to remove a different codeword.
        """
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                # Tentatively remove
                counts[neighbors_arr[c_remove, :]] -= 1
                
                # Check for uncovered words
                uncovered_mask = (counts == 0)
                
                if not np.any(uncovered_mask):
                    # Safe to remove
                    code.pop(i)
                    improved = True
                    break
                else:
                    # We have uncovered words. Can we find a single codeword that covers all of them?
                    # Get the list of uncovered word indices
                    uncovered_words = np.where(uncovered_mask)[0]
                    
                    # Find intersection of reverse_cover for all uncovered words
                    # Start with the first uncovered word's reverse cover
                    if len(uncovered_words) == 0:
                        continue
                        
                    candidates = set(reverse_cover[uncovered_words[0]])
                    
                    # If there are many uncovered words, the intersection might be empty quickly.
                    # Limit the check to a few to avoid overhead, but usually critical sets are small.
                    found_replacement = False
                    for w in uncovered_words[1:]:
                        candidates.intersection_update(reverse_cover[w])
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if candidates:
                        # Pick a random candidate from the intersection
                        replacement = random.choice(list(candidates))
                        # Apply swap
                        counts[neighbors_arr[c_remove, :]] -= 1 # Wait, we already subtracted once above. 
                        # Let's trace:
                        # 1. counts -= neighbors[c_remove]
                        # 2. We found uncovered.
                        # 3. We want to replace c_remove with 'replacement'.
                        # So we need to add back neighbors[c_remove] (undo removal) and subtract neighbors[replacement]?
                        # No, we want to REMOVE c_remove and ADD replacement.
                        # Current state: counts reflects "c_remove removed".
                        # We need to add 'replacement' to the code.
                        counts[neighbors_arr[replacement, :]] += 1
                        code[i] = replacement
                        improved = True
                        break
                    else:
                        # No single replacement covers all uncovered words.
                        # Restore counts and try next codeword
                        counts[neighbors_arr[c_remove, :]] += 1
        
        return code

    def perturb_repair(code, k, deadline):
        """Remove k random codewords and repair greedily."""
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair using greedy logic but starting from existing code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
            
        # While there are uncovered words, add the best candidate
        while time.time() < deadline:
            uncovered = (counts == 0)
            if not np.any(uncovered):
                break
            
            # Gain for each candidate: how many uncovered words it covers
            gains = uncovered[neighbors_arr].sum(axis=1)
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            chosen = random.choice(best_candidates.tolist())
            
            code.append(chosen)
            counts[neighbors_arr[chosen, :]] += 1
        
        return code

    def full_local_search(code, deadline):
        """Combine reduction and local search."""
        code = reduce_code_fast(code)
        code = local_search_improve(code, deadline)
        code = reduce_code_fast(code)
        return code

    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_code = None
    best_size = float('inf')
    
    # 1. Initial Construction
    # Run a few greedy constructions with different seeds to find a good starting point
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    if best_code is None:
        # Fallback if time ran out (should not happen)
        best_code = greedy_construct_vectorized(0)
        best_size = len(best_code)
        
    # 2. Iterated Local Search
    # Alternate between perturbations and local search
    seed = 100
    while time.time() < deadline - 5:
        strategy = seed % 3
        
        if strategy == 0:
            # Small perturbation
            k = random.randint(1, 2)
            perturbed = perturb_repair(best_code, k, deadline - 5)
        elif strategy == 1:
            # Medium perturbation
            k = random.randint(3, 5)
            perturbed = perturb_repair(best_code, k, deadline - 5)
        else:
            # Large perturbation / Fresh start
            # Sometimes a fresh greedy start with a different seed helps escape local minima
            fresh = greedy_construct_vectorized(seed)
            fresh = reduce_code_fast(fresh)
            # Combine with best? Or just use fresh? 
            # Let's try a "merge" approach: take best_code, remove random half, add random half from fresh, repair.
            # For simplicity and speed, let's just do a larger perturbation on best_code
            k = random.randint(6, 10)
            perturbed = perturb_repair(best_code, k, deadline - 5)
            
        # Local search on the perturbed code
        # Use a sub-deadline for local search to ensure we can evaluate and update best
        sub_deadline = min(deadline - 2, time.time() + 5) # Allow up to 5s per local search iteration
        improved = full_local_search(perturbed, sub_deadline)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_code = improved
        
        seed += 1
        
        # Periodic fresh restarts if stuck
        if seed % 10 == 0 and time.time() < deadline - 15:
             fresh = greedy_construct_vectorized(seed)
             fresh = reduce_code_fast(fresh)
             fresh = local_search_improve(fresh, deadline - 10)
             if len(fresh) < best_size:
                 best_size = len(fresh)
                 best_code = fresh

    # Final reduction to ensure no redundancies
    best_code = reduce_code_fast(best_code)
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
