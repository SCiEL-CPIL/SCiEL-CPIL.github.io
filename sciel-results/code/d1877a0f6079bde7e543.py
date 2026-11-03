import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words
        all_words = list(itertools.product(range(q), repeat=n))
        
        # Precompute coverage masks as integers (bitsets)
        # Bit i is set if word i is covered by word j
        # We need dist(w_i, w_j) <= radius
        # For radius 1, w_i is covered by w_j if they differ in 0 or 1 positions.
        
        # Efficient distance calculation for q=3, n=6
        # Convert words to integers or just use tuple comparison? 
        # Numpy is faster for batch, but we need masks for each word.
        
        # Let's stick to numpy for initial mask generation, then convert to int.
        words_np = np.array(all_words)
        
        # dist[i, j] is hamming distance
        # diff = np.abs(words_np[:, None, :] - words_np[None, :, :])
        # dist = (diff != 0).sum(axis=2)
        # coverage_bool = dist <= radius
        
        # This memory usage (729x729) is fine.
        diff = np.abs(words_np[:, None, :] - words_np[None, :, :])
        dist = (diff != 0).sum(axis=2)
        coverage_bool = dist <= radius
        
        # Convert to bitsets
        # Python ints can hold 729 bits easily.
        coverage_masks = [0] * total_words
        for i in range(total_words):
            mask = 0
            # Get indices where coverage_bool[i] is True
            idxs = np.where(coverage_bool[i])[0]
            for idx in idxs:
                mask |= (1 << idx)
            coverage_masks[i] = mask
            
        # Greedy Set Cover using bitsets
        def greedy_set_cover(uncovered_mask, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = uncovered_mask
            code_indices = list(code_indices)
            used = set(code_indices)
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                
                for i in range(total_words):
                    if i in used:
                        continue
                    # Count how many uncovered words i covers
                    new_cover = (coverage_masks[i] & uncovered).bit_count()
                    if new_cover > best_cover_count:
                        best_cover_count = new_cover
                        best_idx = i
                        candidates = [i]
                    elif new_cover == best_cover_count and new_cover > 0:
                        candidates.append(i)
                    
                if best_idx == -1:
                    break
                
                chosen = rng.choice(candidates)
                code_indices.append(chosen)
                used.add(chosen)
                uncovered &= ~coverage_masks[chosen]
                
            return code_indices

        def fast_local_search(code_set):
            code_set = set(code_set)
            # counts[w] is number of codewords covering w
            # Using numpy array for counts is fast for the "min count" check
            counts = np.zeros(total_words, dtype=np.int8)
            
            # Initialize counts
            for c in code_set:
                idxs = np.where(coverage_bool[c])[0]
                counts[idxs] += 1
            
            improved = True
            iterations = 0
            max_iterations = 2000
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning
                removable = None
                for c in list(code_set):
                    # If min count in coverage[c] > 1, c is redundant
                    # coverage_bool[c] is a boolean mask
                    if np.min(counts[coverage_bool[c]]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    idxs = np.where(coverage_bool[removable])[0]
                    counts[idxs] -= 1
                    improved = True
                    continue
                
                # 2. Swaps
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    # Find private words of c
                    # private_mask: True where counts==1 and covered by c
                    private_mask = (counts == 1) & coverage_bool[c]
                    private_words = np.where(private_mask)[0]
                    
                    if len(private_words) == 0:
                        code_set.remove(c)
                        idxs = np.where(coverage_bool[c])[0]
                        counts[idxs] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find candidate w not in code_set that covers all private_words
                    # We want w such that for all p in private_words, dist(w, p) <= 1
                    # Which means w is in intersection of coverage_masks[p] for p in private_words
                    # AND w is not in code_set
                    
                    # Intersection of masks
                    inter_mask = (1 << total_words) - 1 # All bits set
                    for p in private_words:
                        inter_mask &= coverage_masks[p]
                        
                    # Candidates are bits set in inter_mask
                    # We need to check if any candidate is NOT in code_set
                    # code_set_mask
                    code_set_mask = 0
                    for cc in code_set:
                        code_set_mask |= (1 << cc)
                        
                    valid_mask = inter_mask & ~code_set_mask
                    
                    if valid_mask:
                        # Pick a random bit from valid_mask
                        # Get list of indices
                        valid_indices = []
                        m = valid_mask
                        while m:
                            # Find lowest set bit
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            valid_indices.append(idx)
                            m &= m - 1
                            
                        if valid_indices:
                            chosen = random.choice(valid_indices)
                            code_set.remove(c)
                            idxs_c = np.where(coverage_bool[c])[0]
                            counts[idxs_c] -= 1
                            code_set.add(chosen)
                            idxs_new = np.where(coverage_bool[chosen])[0]
                            counts[idxs_new] += 1
                            improved = True
                            swapped = True
                            break
                        
                if not swapped:
                    break
                    
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # Initial Greedy
        # Uncovered is all bits set
        full_mask = (1 << total_words) - 1
        initial_code = greedy_set_cover(full_mask, [])
        ls_code = fast_local_search(initial_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code
            
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 3)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            # Calculate uncovered mask
            covered_mask = 0
            for c in perturbed_code:
                covered_mask |= coverage_masks[c]
            uncovered_mask = full_mask & ~covered_mask
            
            new_code = greedy_set_cover(uncovered_mask, perturbed_code, random_seed=iterations)
            ls_code = fast_local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
        
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
