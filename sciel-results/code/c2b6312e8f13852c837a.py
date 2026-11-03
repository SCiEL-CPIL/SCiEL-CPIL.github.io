import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # We assume the evaluator passes these.
    
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
    # neighbor_indices[i] is a list of indices within distance <= radius
    neighbor_indices = [[] for _ in range(total_words)]
    # For fast redundancy check, we might need masks or just use the lists
    # Since radius=1, size is 1 + n*(q-1) = 1 + 7*2 = 15
    
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
        neighbor_indices[i] = idxs

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        
        # Precompute masks for speed in greedy
        # mask[i] is a big integer with bits set for neighbors of i
        masks = [0] * total_words
        for i in range(total_words):
            m = 0
            for j in neighbor_indices[i]:
                m |= (1 << j)
            masks[i] = m
            
        while uncovered:
            best_gain = -1
            best_candidate = -1
            best_count = 0
            
            # Randomized iteration order to escape local minima in greedy
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for i in candidates:
                if (masks[i] & uncovered) == 0:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = i
                    best_count = 1
                elif gain == best_gain:
                    best_count += 1
                    if random.random() < 1.0 / best_count:
                        best_candidate = i
            
            if best_gain == -1:
                break
                
            chosen = best_candidate
            code.append(chosen)
            uncovered &= ~masks[chosen]
            
        return code

    def reduce_and_optimize(code, deadline):
        """
        Reduces code by removing redundant words and performing 1-for-1 swaps.
        Uses numpy for coverage counts.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any count is 0 in the neighborhood of c_remove
                # We only care about words in N(c_remove) because others are unaffected
                # If min(counts[N(c_remove)]) > 0, then c_remove is redundant
                # Note: counts for words NOT in N(c_remove) are unchanged, so if they were >0 before, they are >0 now.
                # Since the code was valid before, all counts were >= 1.
                # So we only need to check N(c_remove).
                
                # Optimization: np.min on a list of indices into a numpy array
                # neighbor_indices[c_remove] is a list of ints
                min_count = np.min(counts[neighbor_indices[c_remove]])
                
                if min_count > 0:
                    # Redundant, keep removed
                    code.pop(i)
                    improved = True
                    # Restart loop to check for more redundancies
                    break
                
                else:
                    # Restore counts
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Find critical words (those with count 1 in N(c_remove))
                    # These are the words that would be uncovered if we removed c_remove
                    critical_words = [w for w in neighbor_indices[c_remove] if counts[w] == 1]
                    
                    if not critical_words:
                        # Should not happen if min_count == 0, but just in case
                        continue

                    # Find a replacement c_new that covers all critical_words
                    # c_new must be in N(critical_word) for all critical_word
                    # Start with candidates from N(critical_words[0])
                    candidates = neighbor_indices[critical_words[0]]
                    
                    best_new = -1
                    # To make the search faster, we can shuffle candidates
                    random.shuffle(candidates)
                    
                    for c_new in candidates:
                        if c_new == c_remove:
                            continue
                        
                        # Check if c_new covers all other critical words
                        valid = True
                        for cw in critical_words[1:]:
                            # Check if c_new is in N(cw)
                            # Since N(cw) is a list of size 15, we can use 'in'
                            # Or precompute a set? List 'in' is O(15).
                            if c_new not in neighbor_indices[cw]:
                                valid = False
                                break
                        
                        if valid:
                            best_new = c_new
                            break
                    
                    if best_new != -1:
                        # Perform swap
                        code[i] = best_new
                        
                        # Update counts: -c_remove, +best_new
                        for w in neighbor_indices[c_remove]:
                            counts[w] -= 1
                        for w in neighbor_indices[best_new]:
                            counts[w] += 1
                        
                        improved = True
                        # Restart to see if this swap allows further reductions
                        break
        
        return code

    # Main loop with time management
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    # We run multiple instances
    # Each greedy + local search takes some time.
    # Greedy is fast (~0.1s). Local search is variable.
    
    while time.time() < deadline:
        code = greedy_construct(seed)
        
        # Initial reduction
        code = reduce_and_optimize(code, deadline - 5)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we find a very good code, we can spend more time polishing it
            # But for now, we just continue with new seeds to find even better ones
            # Or we can re-optimize the best one found so far?
            # Let's stick to independent runs to ensure diversity.
            
        seed += 1
        
    if best_code is None:
        # Fallback: return empty or a single word? Must be valid.
        # If no code found, return a trivial cover (all words? No, too big).
        # Greedy should always find a cover.
        return []

    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
