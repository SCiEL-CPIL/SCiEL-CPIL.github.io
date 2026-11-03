import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: evaluator kills at 150s
    
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    # Build coverage matrix efficiently using vectorized operations
    # coverage_matrix[c, w] = 1 if word w is within Hamming distance <= radius of codeword c
    coverage_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    
    if radius == 1:
        # Self coverage
        np.fill_diagonal(coverage_matrix, 1)
        
        # For each position and each pair of different values,
        # add coverage from c to w where w differs from c only at that position
        for pos in range(n):
            pow_q = q_powers[pos]
            for v in range(q):
                for v2 in range(q):
                    if v == v2:
                        continue
                    # Words with value v at position pos
                    # Compute indices of all words with value v at pos
                    # word = ... + v * q^pos + ...
                    # We need to find all c where digit[pos] == v, then coverage to c + (v2-v)*q^pos
                    # This is a permutation: c -> c + (v2-v)*q^pos
                    # But we need to only apply where c has digit v at pos
                    
                    # Generate all words with digit v at pos
                    # These are words where (word // q^pos) % q == v
                    # For small total_words, we can just iterate
                    # But let's be smarter:
                    # c ranges over all words where (c // pow_q) % q == v
                    # The set of such c is: for each other combination of digits
                    # Actually, since total_words is only 2187, a simple loop is fine
                    # But let's vectorize:
                    # All words: 0..total_words-1
                    # digit at pos: (np.arange(total_words) // pow_q) % q
                    if not hasattr(construct, '_digit_cache'):
                        construct._digit_cache = {}
                    if pos not in construct._digit_cache:
                        construct._digit_cache[pos] = (np.arange(total_words) // pow_q) % q
                    digits_at_pos = construct._digit_cache[pos]
                    
                    mask_v = (digits_at_pos == v)
                    words_with_v = np.where(mask_v)[0]
                    if len(words_with_v) == 0:
                        continue
                    neighbors = words_with_v + (v2 - v) * pow_q
                    coverage_matrix[words_with_v, neighbors] = 1
    else:
        # Generic radius: for each word c, compute all words within Hamming distance <= radius
        # This is slower but n is small (7) and radius is small (1 for this task)
        # For radius > 1, use the neighbor_indices approach
        neighbor_indices = []
        for i in range(total_words):
            digits = [0] * n
            temp = i
            for k in range(n):
                digits[k] = temp % q
                temp //= q
            
            idxs = [i]
            for r in range(1, radius + 1):
                for positions in product(range(n), repeat=r):
                    if len(set(positions)) < r:
                        continue
                    for changes in product(range(q), repeat=r):
                        if any(changes[k] == digits[positions[k]] for k in range(r)):
                            continue
                        j = i
                        for k in range(r):
                            j += (changes[k] - digits[positions[k]]) * q_powers[positions[k]]
                        if j not in idxs:
                            idxs.append(j)
            
            neighbor_indices.append(idxs)
        
        for c in range(total_words):
            for w in neighbor_indices[c]:
                coverage_matrix[c, w] = 1
    
    # Delete digit cache to free memory
    if hasattr(construct, '_digit_cache'):
        del construct._digit_cache
    
    def greedy_construct(random_seed, temperature=0.0):
        """
        Vectorized greedy construction.
        """
        rng = np.random.RandomState(random_seed)
        is_uncovered = np.ones(total_words, dtype=bool)
        
        code = []
        
        while is_uncovered.any():
            # Compute gain for each candidate
            gains = coverage_matrix @ is_uncovered.astype(np.int32)
            
            # Only consider candidates with gain > 0
            valid_mask = gains > 0
            if not valid_mask.any():
                break
            
            if temperature > 0:
                noise = rng.random(total_words) * temperature
                gains_adjusted = gains.astype(np.float64) + noise
                gains_adjusted[~valid_mask] = -1e18
            else:
                # Add tiny random perturbation to break ties randomly
                noise = rng.random(total_words) * 1e-6
                gains_adjusted = gains.astype(np.float64) + noise
                gains_adjusted[~valid_mask] = -1e18
            
            best_idx = int(np.argmax(gains_adjusted))
            if gains[best_idx] == 0:
                break
            
            code.append(best_idx)
            
            # Update uncovered status
            covered_words = np.where(coverage_matrix[best_idx] == 1)[0]
            is_uncovered[covered_words] = False
        
        return code
    
    def reduce_code(code):
        """
        Remove redundant codewords using coverage counting.
        """
        if not code:
            return code
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            coverage_count += coverage_matrix[c]
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                coverage_count -= coverage_matrix[c]
                if np.any(coverage_count == 0):
                    coverage_count += coverage_matrix[c]
                    new_code.append(c)
                else:
                    changed = True
            
            code = new_code
        
        return code
    
    def local_search_swap(code, max_iterations=50):
        """
        Swap-based local search: try replacing codewords with better alternatives.
        """
        if not code or time.time() > deadline - 10.0:
            return code
        
        code_list = list(code)
        code_set = set(code_list)
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            coverage_count += coverage_matrix[c]
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10.0:
            improved = False
            iteration += 1
            
            # Try removing each codeword
            for i in range(len(code_list)):
                c = code_list[i]
                coverage_count -= coverage_matrix[c]
                if np.any(coverage_count == 0):
                    coverage_count += coverage_matrix[c]
                else:
                    code_list.pop(i)
                    code_set.discard(c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: for each codeword, try replacing with a random word
            for _ in range(30):
                if time.time() > deadline - 10.0:
                    break
                if not code_list:
                    break
                
                i = random.randint(0, len(code_list) - 1)
                new_word = random.randint(0, total_words - 1)
                
                if new_word in code_set:
                    continue
                
                old_word = code_list[i]
                
                # Check if swap is valid
                coverage_count -= coverage_matrix[old_word]
                coverage_count += coverage_matrix[new_word]
                
                if np.any(coverage_count == 0):
                    coverage_count -= coverage_matrix[new_word]
                    coverage_count += coverage_matrix[old_word]
                    continue
                
                # Swap is valid; accept it
                code_list[i] = new_word
                code_set.discard(old_word)
                code_set.add(new_word)
                improved = True
                
                # Try to remove redundancies after swap
                changed = True
                while changed and time.time() < deadline - 10.0:
                    changed = False
                    for j in range(len(code_list)):
                        c = code_list[j]
                        coverage_count -= coverage_matrix[c]
                        if np.any(coverage_count == 0):
                            coverage_count += coverage_matrix[c]
                        else:
                            code_list.pop(j)
                            code_set.discard(c)
                            changed = True
                            break
        
        return code_list
    
    # Main search loop
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy constructions with different seeds
    seeds_to_try = [0, 1, 42, 123, 456, 789, 1000, 2024, 3141, 2718]
    
    for seed in seeds_to_try:
        if time.time() > deadline - 20.0:
            break
        
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    # Apply local search to the best code found
    if best_code is not None and time.time() < deadline - 25.0:
        ls_code = local_search_swap(best_code, max_iterations=100)
        ls_code = reduce_code(ls_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code[:]
    
    # Try more greedy restarts if time permits
    extra_seed = 100
    while time.time() < deadline - 15.0:
        code = greedy_construct(extra_seed)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            # Apply local search to the new best
            if time.time() < deadline - 10.0:
                ls_code = local_search_swap(best_code, max_iterations=50)
                ls_code = reduce_code(ls_code)
                if len(ls_code) < best_size:
                    best_size = len(ls_code)
                    best_code = ls_code[:]
        
        extra_seed += 1
    
    # Final local search on best code
    if best_code is not None and time.time() < deadline - 5.0:
        ls_code = local_search_swap(best_code, max_iterations=200)
        ls_code = reduce_code(ls_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code[:]
    
    # Convert best_code indices to list of lists of digits
    if best_code is None:
        # Fallback: return empty or a single word
        best_code = [0]
    
    result = []
    seen = set()
    for idx in best_code:
        if idx in seen:
            continue
        seen.add(idx)
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % q
            temp //= q
        result.append(digits)
    
    return result
