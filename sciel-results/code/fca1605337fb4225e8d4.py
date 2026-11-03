import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distance matrix
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # coverage[i] is a set of indices j such that dist[i][j] <= radius
        coverage = [set() for _ in range(total_words)]
        # coverers[j] is a list of indices i such that j in coverage[i]
        coverers = [[] for _ in range(total_words)]
        
        for i in range(total_words):
            idxs = np.where(dist[i] <= radius)[0]
            coverage[i] = set(idxs.tolist())
            for j in idxs:
                coverers[j].append(i)
                
        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = set(uncovered)
            code_indices = list(code_indices)
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                # Optimization: Only consider candidates that cover at least one uncovered word?
                # We can iterate all, but check intersection size.
                for i in range(total_words):
                    new_cover = len(coverage[i] & uncovered)
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
                uncovered -= coverage[chosen]
                
            return code_indices

        def local_search(code_set):
            code_set = set(code_set)
            
            # counts[w] is number of codewords in code_set covering w
            counts = np.zeros(total_words, dtype=int)
            for c in code_set:
                counts[list(coverage[c])] += 1
            
            improved = True
            iterations = 0
            max_iterations = 500 # Increase iterations for stronger search
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning (Deletion)
                # Find a codeword c where all words in coverage[c] are covered by >1 codeword
                removable = -1
                for c in list(code_set):
                    # Check if min count in coverage[c] > 1
                    # Using numpy is fast
                    if np.min(counts[list(coverage[c])]) > 1:
                        removable = c
                        break
                        
                if removable != -1:
                    code_set.remove(removable)
                    counts[list(coverage[removable])] -= 1
                    improved = True
                    continue
                
                # 2. Swaps (1-swap)
                # Try to swap c with c'
                # For each c in code_set:
                #   Find private words P = {w in coverage[c] | counts[w] == 1}
                #   If P is empty, delete c (handled above, but maybe counts changed? No, we break on delete)
                #   Find c' not in code_set such that P subset coverage[c']
                #   We do this by intersecting coverers[w] for w in P
                
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = [w for w in coverage[c] if counts[w] == 1]
                    
                    if not private_words:
                        # This case is handled by pruning, but if pruning failed to catch it 
                        # due to order? Pruning checks min count > 1. 
                        # If min count == 1, there is a private word.
                        # So if private_words is empty, min count > 1, so it should have been pruned.
                        # Unless we are in the middle of a loop and counts changed? 
                        # We break after prune. So this shouldn't happen.
                        continue
                    
                    # Find candidates c' that cover all private words
                    # Start with all codewords, intersect with coverers of each private word
                    # Optimization: Start with the smallest coverers list
                    private_words_sorted = sorted(private_words, key=lambda w: len(coverers[w]))
                    candidates = set(coverers[private_words_sorted[0]])
                    
                    for w in private_words_sorted[1:]:
                        candidates &= set(coverers[w])
                        if not candidates:
                            break
                    
                    # Remove current code_set from candidates
                    candidates -= code_set
                    
                    if candidates:
                        # Pick a candidate (randomly to diversify)
                        c_new = random.choice(list(candidates))
                        code_set.remove(c)
                        counts[list(coverage[c])] -= 1
                        code_set.add(c_new)
                        counts[list(coverage[c_new])] += 1
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
        
        # Initial Greedy + Local Search
        initial_code = greedy_set_cover(set(range(total_words)), [])
        ls_code = local_search(initial_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code
            
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            # Perturb
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 3) # Increase perturbation
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            covered = set()
            for c in perturbed_code:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            ls_code = local_search(new_code)
            
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
