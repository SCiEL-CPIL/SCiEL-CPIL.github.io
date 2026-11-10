"""
Ternary covering code K_3(7,1) - ILS with targeted repair and local search removal.
"""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Find a covering code of minimum size for {0..q-1}^n with covering radius.
    For the specific task: q=3, n=7, radius=1.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        # Use direct sum construction as fallback
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force small case
            words = list(itertools.product(range(q), repeat=n))
            random.shuffle(words)
            return words
    
    # Specific case: q=3, n=7, radius=1
    # Precompute neighbors for all 2187 words
    # Each word has 1 + 6*2 = 13 neighbors in the ball of radius 1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor lists for each word (the ball of radius 1)
    # For word w, neighbors are w itself and words that differ in exactly one position
    neighbor_lists = []
    for w in all_words:
        w = list(w)
        neighbors = [w]
        for pos in range(7):
            old = w[pos]
            for v in range(3):
                if v != old:
                    w[pos] = v
                    neighbors.append(tuple(w))
            w[pos] = old
        neighbor_lists.append(neighbors)
    
    # Precompute for each word index, the list of word indices in its ball
    ball_indices = []
    for neighbors in neighbor_lists:
        ball_indices.append([word_to_idx[ne] for ne in neighbors])
    
    # Greedy initialization: start with a large random set, then remove
    def greedy_init():
        """Create an initial covering code using greedy heuristic."""
        # Start with all 2187 words
        code_set = set(range(2187))
        # Count how many codewords cover each word
        coverage_count = np.zeros(2187, dtype=np.int32)
        for c in code_set:
            for nb in ball_indices[c]:
                coverage_count[nb] += 1
        
        # Remove words that are still covered by other codewords
        # Process in random order for diversity
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in code_set:
                continue
            # Check if removing c leaves all its neighbors covered
            can_remove = True
            for nb in ball_indices[c]:
                if coverage_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove c
                code_set.remove(c)
                for nb in ball_indices[c]:
                    coverage_count[nb] -= 1
        
        return list(code_set)
    
    def is_covered(code_set):
        """Check if the code covers all words."""
        covered = np.zeros(2187, dtype=np.bool_)
        for c in code_set:
            for nb in ball_indices[c]:
                covered[nb] = True
        return np.all(covered)
    
    def get_uncovered(code_set):
        """Return list of uncovered word indices."""
        covered = np.zeros(2187, dtype=np.bool_)
        for c in code_set:
            for nb in ball_indices[c]:
                covered[nb] = True
        return np.where(~covered)[0]
    
    def local_search_remove(code_list):
        """Try to remove words from the code while maintaining coverage."""
        code_set = set(code_list)
        coverage_count = np.zeros(2187, dtype=np.int32)
        for c in code_set:
            for nb in ball_indices[c]:
                coverage_count[nb] += 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            candidates = list(code_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if c not in code_set:
                    continue
                # Check if removing c is safe
                can_remove = True
                for nb in ball_indices[c]:
                    if coverage_count[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    code_set.remove(c)
                    for nb in ball_indices[c]:
                        coverage_count[nb] -= 1
                    removed_any = True
        
        return list(code_set)
    
    def repair(code_set, uncovered):
        """Add words to cover uncovered positions."""
        if not uncovered:
            return code_set, 0
        
        added = 0
        uncovered_set = set(uncovered)
        
        # For each uncovered word, find the codeword candidate that covers the most uncovered words
        # We'll add one at a time, choosing the best candidate each time
        while uncovered_set:
            best_cand = None
            best_count = 0
            # Sample a subset of candidates to speed up
            candidates = list(range(2187))
            random.shuffle(candidates)
            for cand in candidates[:500]:  # Only check 500 random candidates
                # Count how many uncovered words this candidate covers
                count = 0
                for nb in ball_indices[cand]:
                    if nb in uncovered_set:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_cand = cand
            
            if best_cand is not None and best_count > 0:
                code_set.add(best_cand)
                added += 1
                # Update uncovered set
                for nb in ball_indices[best_cand]:
                    if nb in uncovered_set:
                        uncovered_set.remove(nb)
        
        return code_set, added
    
    # ILS: Iterated Local Search
    best_size = 10**9
    best_code = None
    
    # Initial solution
    current_code = greedy_init()
    current_code = local_search_remove(current_code)
    best_size = len(current_code)
    best_code = current_code.copy()
    
    # ILS loop
    while time.time() < deadline:
        # Perturbation: randomly add some words
        perturbed = list(current_code)
        n_add = random.randint(3, 10)
        for _ in range(n_add):
            rand_word = random.randint(0, 2187 - 1)
            if rand_word not in perturbed:
                perturbed.append(rand_word)
        
        # Local search: remove
        perturbed = local_search_remove(perturbed)
        
        # Check if better
        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_code = perturbed.copy()
            current_code = perturbed.copy()
        else:
            # Accept with some probability (Simulated Annealing-like)
            # or just keep current
            if random.random() < 0.3:
                current_code = perturbed.copy()
        
        # Additional local search pass
        current_code = local_search_remove(current_code)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
    
    # Final local search on best
    best_code = local_search_remove(best_code)
    
    # Verify coverage
    uncovered = get_uncovered(best_code)
    if uncovered:
        # Repair if needed
        best_code_set = set(best_code)
        best_code_set, _ = repair(best_code_set, uncovered)
        best_code = list(best_code_set)
    
    # Convert to list of codewords
    result = [list(all_words[i]) for i in best_code]
    return result
