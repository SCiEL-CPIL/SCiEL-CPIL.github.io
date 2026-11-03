import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But the function signature is generic. We assume the evaluator passes these.
    # We will optimize for the specific case q=3, n=7, radius=1 but keep it general enough.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: simple greedy
        # For now, focus on the specific task instance as per instructions
        total_words = q ** n
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
        
        # Precompute neighbors
        neighbors_mask = [0] * total_words
        for i in range(total_words):
            w = idx_to_tuple(i)
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    j = tuple_to_idx(tuple(new_w))
                    mask |= (1 << j)
            neighbors_mask[i] = mask
        
        def greedy_construct(random_seed=None):
            if random_seed is not None:
                random.seed(random_seed)
            uncovered = (1 << total_words) - 1
            code = []
            while uncovered:
                best_gain = -1
                best_idx = -1
                candidates = list(range(total_words))
                random.shuffle(candidates)
                for i in candidates:
                    if (neighbors_mask[i] & uncovered) == 0:
                        continue
                    gain = (neighbors_mask[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                code.append(best_idx)
                uncovered &= ~neighbors_mask[best_idx]
            return code
        
        def idx_to_list(idx):
            return list(idx_to_tuple(idx))
        
        code = greedy_construct(0)
        return [idx_to_list(i) for i in code]

    total_words = q ** n
    
    # Precompute neighbors as bitmasks and index lists
    # Map word tuple to index
    # Index = sum(digit * q^i)
    
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

    # Precompute neighbors
    neighbors_mask = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 0
        idxs = [i]
        # Distance 0
        mask |= (1 << i)
        # Distance 1
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
                idxs.append(j)
        neighbors_mask[i] = mask
        neighbor_indices[i] = idxs

    def greedy_construct(random_seed, strategy=0):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # To add randomness, we can pick from top few or just randomize iteration order
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for i in candidates:
                if (neighbors_mask[i] & uncovered) == 0:
                    continue
                gain = (neighbors_mask[i] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == -1:
                break # Should not happen if uncovered is not 0 and masks are correct
                
            # Pick randomly from the best candidates to introduce diversity
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered &= ~neighbors_mask[chosen]
            
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Try removing c
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Restore
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    # Keep removed
                    changed = True
            
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """
        Fast local search using coverage counts.
        1. Try to remove a codeword.
        2. If removal creates uncovered words, try to replace it with a better one
           that covers the gap and potentially allows further reduction.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid getting stuck in specific order
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Restore
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Find the best replacement
                    # We need to cover the words that are now at count 0 (or would be if we didn't restore)
                    # Actually, we restored, so we need to identify which words were at risk.
                    # Let's re-calculate the "uncovered" set if we were to remove c_remove.
                    # Since we restored, we can just find words w where counts[w] == 1 and w is in neighbor_indices[c_remove].
                    # These are the critical words.
                    critical_words = []
                    for w in neighbor_indices[c_remove]:
                        if counts[w] == 1:
                            critical_words.append(w)
                    
                    if not critical_words:
                        # This shouldn't happen if np.any(counts==0) was true, but just in case
                        continue

                    # We need a word w_new such that neighbors_mask[w_new] covers all critical_words?
                    # Or at least covers enough to allow removal of another codeword?
                    # Simple heuristic: Find w_new that maximizes coverage of critical_words.
                    # If it covers all, we can swap.
                    # If it covers some, we might still swap if it allows removing another codeword later?
                    # Let's stick to: Find w_new that covers ALL critical_words.
                    # If found, swap c_remove with w_new.
                    # Then run a reduction pass.
                    
                    best_w = -1
                    # Optimization: Only check words that cover at least one critical word.
                    # Precompute which words cover which critical words? Too expensive.
                    # Just iterate all words? 2187 is small.
                    for w in range(total_words):
                        if w == c_remove:
                            continue
                        # Check if w covers all critical_words
                        # We can use bitmasks for this check faster?
                        # critical_mask = 0
                        # for cw in critical_words: critical_mask |= (1 << cw)
                        # if (neighbors_mask[w] & critical_mask) == critical_mask:
                        
                        # Let's just loop critical_words
                        covers_all = True
                        for cw in critical_words:
                            if not (neighbors_mask[w] & (1 << cw)):
                                covers_all = False
                                break
                        if covers_all:
                            # Check if this replacement is beneficial
                            # It is beneficial if w_new covers more uncovered words than c_remove did?
                            # Or simply if it allows a reduction?
                            # Let's just accept the first valid replacement to keep it simple and fast.
                            # Better: prefer one that is "better" in some way?
                            # For now, accept.
                            best_w = w
                            break
                    
                    if best_w != -1:
                        # Perform swap
                        code[i] = best_w
                        # Update counts
                        for w in neighbor_indices[best_w]:
                            counts[w] += 1
                        improved = True
                        # After swap, try to reduce again?
                        # The outer loop will handle this by re-shuffling and trying again.
                        # But we can immediately try to remove others?
                        # Let's just continue the loop.
                else:
                    # Removal was safe
                    # Remove c_remove from code
                    code.pop(i)
                    # Update counts (already decremented)
                    improved = True
                    # Break out of inner loop to restart with new code size
                    break
        
        return code

    # Main loop with time management
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code_fast(code)
        
        # Local search improvement
        code = local_search_improve(code, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
        # If we hit 186, we can stop? No, lower bound is 156. 
        # But 186 is best known. If we find 186, it's a great result.
        if best_size <= 186:
            break

    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
