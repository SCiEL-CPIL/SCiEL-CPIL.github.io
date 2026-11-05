import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute spheres as Python integers (bitsets) and neighbor lists
    sphere_bits = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    
    def word_to_idx(w):
        idx = 0
        for k in range(n):
            idx = idx * q + w[k]
        return idx

    # Precompute spheres
    for i, word in enumerate(all_words):
        sphere = set([i])
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word_list = list(word)
                    new_word_list[pos] = val
                    new_word = tuple(new_word_list)
                    j = word_to_idx(new_word)
                    sphere.add(j)
        else:
            # Generic radius
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for offsets in itertools.product(range(q), repeat=dist):
                        if all(word[p] == o for p, o in zip(positions, offsets)):
                            continue
                        new_word_list = list(word)
                        for pos, off in zip(positions, offsets):
                            new_word_list[pos] = off
                        new_word = tuple(new_word_list)
                        j = word_to_idx(new_word)
                        sphere.add(j)
        
        bits = 0
        for j in sphere:
            bits |= (1 << j)
            neighbor_indices[i].append(j)
        sphere_bits[i] = bits

    full_mask = (1 << total_words) - 1
    best_solution = None
    best_size = total_words + 1
    
    start_time = time.time()
    max_time = 115
    
    # Use multiple restarts with randomized tie-breaking
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        covered_bits = 0
        selected = []
        selected_set = set()
        
        # Randomize the order of selection in case of ties?
        # Or just standard greedy.
        # To get diversity, we can shuffle the initial candidate order or pick randomly among top K.
        # Let's try standard greedy first, it's fast.
        
        while True:
            if covered_bits == full_mask:
                break
            
            best_gain = -1
            best_idx = -1
            uncovered_bits = (~covered_bits) & full_mask
            
            # Iterate through all words to find the one covering the most uncovered words
            # To add diversity in later restarts, we can add a small random weight
            # or just rely on the fact that different starting points might be explored?
            # Actually, greedy is deterministic. 
            # To make restarts useful, we can randomly select the FIRST word, 
            # or randomly break ties.
            
            # Let's randomly break ties:
            # Collect all with max gain? That's expensive.
            # Instead, let's just run standard greedy. 
            # But standard greedy always produces the same result!
            # So we MUST introduce randomness.
            
            # Strategy: Randomly select the first codeword.
            if not selected:
                best_idx = random.randint(0, total_words - 1)
                selected.append(best_idx)
                selected_set.add(best_idx)
                covered_bits |= sphere_bits[best_idx]
                continue
            
            # For subsequent steps, we can also randomize tie-breaking.
            # Let's find the max gain, and if there are many, pick one randomly.
            # To do this efficiently, we can just iterate and keep a list of max gainers?
            # Or just pick the first one found?
            # Let's pick the first one found to keep it fast, but the first word is random.
            
            for i in range(total_words):
                if i in selected_set:
                    continue
                gain = (sphere_bits[i] & uncovered_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            covered_bits |= sphere_bits[best_idx]
            
        # Local Search: Redundancy Removal
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            for nb in neighbor_indices[c]:
                cov_count[nb] += 1
        
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                is_redundant = True
                for nb in neighbor_indices[c]:
                    if cov_count[nb] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for nb in neighbor_indices[c]:
                        cov_count[nb] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = list(selected)
            
    return [all_words[i] for i in best_solution]
