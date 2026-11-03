import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But the function signature is generic. We assume the evaluator passes these.
    # We will optimize for the specific case q=3, n=7, radius=1 but keep it general enough.
    
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
                
                # Strategy 4: Weighted greedy using digit rarity as tiebreaker
                # If gains are equal, prefer codewords with "rarer" digits?
                # This is a heuristic. Let's stick to standard gain first, but maybe add noise.
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
        counts = np.zeros(total_words, dtype=np.int32)
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
        Try to swap codewords to reduce the total number.
        1. Remove a codeword.
        2. If uncovered words appear, try to replace the removed codeword with a different one 
           that covers more of the newly uncovered set or allows removing another codeword.
        This is a simplified 1-swap local search.
        """
        code = list(code)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each codeword
            for i in range(len(code)):
                if time.time() >= deadline:
                    break
                c_remove = code[i]
                # Tentatively remove
                new_code = [c for j, c in enumerate(code) if j != i]
                
                # Check coverage
                covered = 0
                for c in new_code:
                    covered |= neighbors_mask[c]
                
                uncovered = ((1 << total_words) - 1) & ~covered
                
                if uncovered == 0:
                    # We removed a redundant codeword
                    code = new_code
                    improved = True
                    break # Restart loop with new code
                
                # If not fully covered, try to find a replacement
                # We need a word w such that neighbors_mask[w] covers the uncovered set?
                # Or at least covers more uncovered than c_remove did?
                # Actually, we want to replace c_remove with w such that the union of masks of new_code + {w}
                # covers the uncovered set, and ideally we can then remove another codeword.
                
                # Simple heuristic: Find w that maximizes coverage of `uncovered`.
                best_w = -1
                best_gain = 0
                # Only check words that cover at least one uncovered word
                # Iterate all words? 2187 is small.
                for w in range(total_words):
                    if w == c_remove:
                        continue
                    # Check if w covers any uncovered
                    gain = (neighbors_mask[w] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                
                if best_w != -1 and best_gain > 0:
                    # Tentatively replace
                    test_code = new_code + [best_w]
                    
                    # Check if this new code is valid (covers everything)
                    covered_test = 0
                    for c in test_code:
                        covered_test |= neighbors_mask[c]
                    
                    if (covered_test & ((1 << total_words) - 1)) == ((1 << total_words) - 1):
                        # Valid replacement. 
                        # Now, can we remove any other codeword from test_code?
                        # Run a quick reduction pass on test_code
                        reduced = reduce_code_fast(test_code)
                        if len(reduced) < len(code):
                            code = reduced
                            improved = True
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
        
        # Try local search improvement if time permits
        if len(code) < best_size or (best_code is None):
            if time.time() < deadline - 5: # Leave time for local search
                code = local_search_improve(code, deadline - 2)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we hit 186, we can stop? No, lower bound is 156. 
            # But 186 is best known. If we find 186, it's a great result.
            if best_size <= 186:
                # Keep searching a bit more to see if we can go lower
                pass
                # break # Don't break, try to beat 186
        
        seed += 1
        
    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
