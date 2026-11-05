import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # We only care about the specific case q=3, n=7, radius=1 for this task.
    # If parameters differ, fallback to a simple construction or raise error.
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this specific task evaluation)
        # Simple greedy for general case
        return _general_greedy(q, n, radius, start_time, time_limit)
    
    # Specific optimization for K_3(7,1)
    total_words = q ** n
    
    # Precompute neighbors for all 2187 words using NumPy for efficiency
    # Each word is a 7-digit base-3 number
    # neighbor_masks[i] is an integer bitset where bit j is 1 if word j is within distance 1 of word i
    
    # Generate all words as a 2D array of shape (2187, 7)
    words = np.array([[idx // (q ** (n - 1 - i)) % q for i in range(n)] for idx in range(total_words)])
    
    # Precompute neighbor masks using NumPy vectorization
    # For each word, compute all distance-1 neighbors
    neighbor_masks = np.zeros(total_words, dtype=object)
    
    for i in range(total_words):
        mask = 1 << i
        w = words[i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Find index of new_w
                new_idx = 0
                for digit in reversed(new_w):
                    new_idx = new_idx * q + digit
                mask |= (1 << new_idx)
        neighbor_masks[i] = mask
    
    # Convert to list for faster access in Python loops
    neighbor_masks_list = neighbor_masks.tolist()
    
    best_size = float('inf')
    best_result = None
    
    # Spectral-inspired initialization: use the structure of the Hamming graph
    # The Hamming graph H(7,3) is vertex-transitive, so we can use symmetry
    # to create structured initial covers
    
    def spectral_greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        chosen = []
        
        # Create candidate ordering with spectral-inspired bias
        # Words with lower "spectral coordinate" first
        # Approximate spectral coordinates using position in the Hamming graph
        
        candidates = list(range(total_words))
        
        # Sort by a heuristic based on word structure
        # Words with more 0s or more 2s might be better "centers"
        def spectral_score(idx):
            w = words[idx]
            # Sum of positions weighted by value
            return sum(pos * w[pos] for pos in range(n))
        
        # Partially sort with randomness
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (spectral_score(x) + random.random() * 10) % 20)
        
        while uncovered:
            # Vectorized gain computation
            # Compute gain for all candidates at once using NumPy
            gains = np.array([(neighbor_masks_list[idx] & uncovered).bit_count() for idx in candidates], dtype=np.int32)
            
            best_idx = candidates[np.argmax(gains)]
            best_gain = gains.max()
            
            if best_gain == 0:
                break
                
            chosen.append(best_idx)
            uncovered &= ~neighbor_masks_list[best_idx]
            
            # Periodically re-sort candidates to maintain good ordering
            if len(chosen) % 20 == 0:
                candidates.sort(key=lambda x: (neighbor_masks_list[x] & uncovered).bit_count() + random.random(), reverse=True)
        
        return chosen
    
    def local_search_with_exchanges(initial_cover, time_budget=5.0):
        """Local search with 1-for-0 removals and 2-for-1 exchanges"""
        chosen = list(initial_cover)
        start_ls = time.time()
        
        # Precompute coverage for each codeword
        def get_coverage(codeword_idx):
            return neighbor_masks_list[codeword_idx]
        
        # First pass: remove redundant codewords
        pruned = True
        while pruned:
            pruned = False
            for i in range(len(chosen)):
                # Check if chosen[i] is redundant
                temp_union = 0
                for j in range(len(chosen)):
                    if i != j:
                        temp_union |= neighbor_masks_list[chosen[j]]
                
                if temp_union == (1 << total_words) - 1:
                    chosen.pop(i)
                    pruned = True
                    break
        
        # Second pass: try 2-for-1 exchanges
        improved = True
        while improved and (time.time() - start_ls) < time_budget:
            improved = False
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    # Try removing chosen[i] and chosen[j]
                    remaining = [c for k, c in enumerate(chosen) if k != i and k != j]
                    
                    # Compute coverage of remaining
                    temp_union = 0
                    for c in remaining:
                        temp_union |= neighbor_masks_list[c]
                    
                    # Find uncovered words
                    uncovered = (1 << total_words) - 1 - temp_union
                    
                    if uncovered == 0:
                        # Both can be removed
                        chosen = remaining
                        improved = True
                        break
                    
                    # Find if there's a single codeword that covers all uncovered
                    # among words not in chosen
                    candidate_replacements = []
                    for c in range(total_words):
                        if c in chosen:
                            continue
                        if (neighbor_masks_list[c] & uncovered) == uncovered:
                            candidate_replacements.append(c)
                    
                    if candidate_replacements:
                        # Replace i and j with the best candidate
                        # Pick the one that covers most additional covered words
                        best_c = max(candidate_replacements, key=lambda c: (neighbor_masks_list[c] & temp_union).bit_count())
                        new_chosen = remaining + [best_c]
                        if len(new_chosen) < len(chosen):
                            chosen = new_chosen
                            improved = True
                            break
                
                if improved:
                    break
        
        return chosen
    
    # Run multiple iterations with different strategies
    num_iterations = 200
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Phase 1: Greedy cover
        cover = spectral_greedy_cover(random_seed=i)
        
        if not cover:
            continue
        
        # Phase 2: Local search
        refined = local_search_with_exchanges(cover, time_budget=2.0)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_result = refined.copy()
            
            if best_size <= 156:  # Theoretical lower bound
                break
    
    # Convert indices back to words
    if best_result:
        return [list(words[idx]) for idx in best_result]
    else:
        # Fallback to product construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _general_greedy(q, n, radius, start_time, time_limit):
    # Fallback for other parameters
    # Simple greedy
    total_words = q ** n
    
    # Generate all words
    words = []
    for idx in range(total_words):
        w = []
        rem = idx
        for i in range(n):
            w.append(rem % q)
            rem //= q
        w.reverse()
        words.append(tuple(w))
    
    # Precompute neighbors
    neighbor_masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w_tuple = tuple(new_w)
                # Find index
                new_idx = 0
                for digit in reversed(new_w):
                    new_idx = new_idx * q + digit
                mask |= (1 << new_idx)
        neighbor_masks[i] = mask
    
    # Greedy set cover
    uncovered = (1 << total_words) - 1
    chosen = []
    candidates = list(range(total_words))
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        for idx in candidates:
            gain = (neighbor_masks[idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx == -1:
            break
        
        chosen.append(best_idx)
        uncovered &= ~neighbor_masks[best_idx]
    
    return [list(words[idx]) for idx in chosen]

# EVOLVE-BLOCK-END
