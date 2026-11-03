import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius, start_time, deadline)
    
    # Specific optimization for q=3, n=6, r=1
    total_words = 3 ** n  # 729
    
    # Helper functions
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * 3 + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute coverage masks for each word
    # cov_masks[i] is a 729-bit integer representing the set of words covered by word i
    cov_masks = [0] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        mask = 1 << idx  # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                nb_idx = word_to_idx(tuple(new_w_list))
                mask |= (1 << nb_idx)
        cov_masks[idx] = mask
    
    all_mask = (1 << total_words) - 1

    # 1. Generate Tetracode-based initial cover (size 81)
    # Tetracode is K_3(4,1) with 9 codewords:
    # (a, b, a+b, a+2b) mod 3 for a,b in 0..2
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
    
    # Direct sum with K_3(2,0) which is just {0,1,2} for each of the last 2 positions
    # Actually, direct sum of C1 (len 4) and C2 (len 2) creates codes of len 6.
    # C2 for radius 0 is the full space {0,1,2}^2 (size 9).
    initial_code = []
    for t in tetracode:
        for x in range(3):
            for y in range(3):
                w = t + (x, y)
                initial_code.append(word_to_idx(w))
                
    # 2. Helper to reduce cover size by removing redundant codewords
    def reduce_cover(code_list):
        code_set = set(code_list)
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry and ensure all get a chance
            candidates = list(code_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in code_set:
                    continue
                # Try removing c
                code_set.remove(c)
                # Check coverage
                cov = 0
                for cc in code_set:
                    cov |= cov_masks[cc]
                if cov == all_mask:
                    improved = True
                    # Restart the loop with the new set to catch cascading removals
                    break
                else:
                    code_set.add(c)
        return list(code_set)

    # 3. Initial reduction
    best_code = reduce_cover(initial_code)
    best_size = len(best_code)
    
    # 4. Local Search with Perturbation
    # If we are stuck at > 73, try swapping words to create opportunities for removal
    while time.time() < deadline - 10.0:
        if best_size <= 73:
            break
            
        # Perturb: pick 1 to 3 random codewords and replace them with random others
        perturbed = list(best_code)
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            if not perturbed:
                break
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            removed_cw = perturbed[idx_to_remove]
            
            # Pick a random replacement not currently in the set
            while True:
                new_cw = random.randint(0, total_words - 1)
                if new_cw not in perturbed:
                    break
            
            perturbed[idx_to_remove] = new_cw
        
        # Repair: Reduce the perturbed cover
        candidate_code = reduce_cover(perturbed)
        
        if len(candidate_code) < best_size:
            best_size = len(candidate_code)
            best_code = candidate_code
            # If we hit 73, we can try a few more iterations to see if 72 is possible,
            # but 73 is the known optimum, so we can stop early to save time for safety.
            if best_size == 73:
                # Run a few more quick iterations to try for 72?
                # Given time constraints and 71 lower bound, 72 is rare.
                # Let's break to ensure we return a valid 73 quickly.
                break

    # 5. Final verification
    final_cov = 0
    for c in best_code:
        final_cov |= cov_masks[c]
    
    if final_cov != all_mask:
        # Fallback: Return the initial reduced tetracode or a greedy cover
        # This shouldn't happen if reduce_cover is correct, but just in case
        # Re-run reduce on initial
        best_code = reduce_cover(initial_code)
        # If still broken (shouldn't be), fallback to greedy
        if final_cov != all_mask:
             # Simple greedy fallback
             uncovered = all_mask
             greedy_code = []
             candidates = list(range(total_words))
             random.shuffle(candidates)
             while uncovered != 0:
                 best_cand = -1
                 best_gain = 0
                 for cand in candidates:
                     gain = (cov_masks[cand] & uncovered).bit_count()
                     if gain > best_gain:
                         best_gain = gain
                         best_cand = cand
                 if best_cand == -1 or best_gain == 0:
                     break
                 greedy_code.append(best_cand)
                 uncovered &= ~cov_masks[best_cand]
             best_code = greedy_code

    return [list(idx_to_word(cw)) for cw in best_code]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback implementation for other parameters
    total_words = q ** n
    if total_words > 10000:
        return []
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors
    neighbors = {}
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbs = {idx}
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = new_val
                nbs.add(word_to_idx(tuple(nw)))
        neighbors[idx] = nbs
    
    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_cand = -1
        best_gain = 0
        for cand in list(uncovered) + list(random.sample(range(total_words), min(100, total_words))):
            gain = len(neighbors[cand] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand == -1 or best_gain == 0:
            break
        code.append(best_cand)
        uncovered -= neighbors[best_cand]
    
    return [list(idx_to_word(cw)) for cw in code]
