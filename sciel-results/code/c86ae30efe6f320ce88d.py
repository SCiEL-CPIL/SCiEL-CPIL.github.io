import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Pre-compute spheres
    spheres = []
    for i, word in enumerate(all_words):
        sphere = [i]
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word_list = list(word)
                    new_word_list[pos] = val
                    new_word = tuple(new_word_list)
                    sphere.append(word_to_idx[new_word])
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for offsets in itertools.product(range(1, q), repeat=dist):
                        new_word_list = list(word)
                        for pos, off in zip(positions, offsets):
                            new_word_list[pos] = (word[pos] + off) % q
                        new_word = tuple(new_word_list)
                        sphere.append(word_to_idx[new_word])
        spheres.append(sphere)
        
    def update_cc(cc, c_idx, delta):
        for idx in spheres[c_idx]:
            cc[idx] += delta

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=np.bool_)
    selected = []
    cover_count = np.zeros(N, dtype=np.int32)
    
    while uncovered.any():
        counts = np.array([np.sum(uncovered[spheres[i]]) for i in range(N)])
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        selected.append(best_idx)
        update_cc(cover_count, best_idx, 1)
        uncovered[spheres[best_idx]] = False

    # 2. Local Search
    def remove_redundant():
        nonlocal selected, cover_count
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                min_cov = np.min(cover_count[spheres[c]])
                if min_cov > 1:
                    update_cc(cover_count, c, -1)
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

    remove_redundant()
    
    # 3. Iterated Swap Search
    start_time = time.time()
    max_time = 100
    
    while time.time() - start_time < max_time:
        improved = False
        
        for _ in range(50):
            if not selected: break
            c = random.choice(selected)
            
            crit_words = [w for w in spheres[c] if cover_count[w] == 1]
            
            if not crit_words:
                continue
                
            c_new = random.randint(0, N - 1)
            if c_new in selected:
                continue
                
            valid = True
            for w in crit_words:
                if w not in spheres[c_new]:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Save State
            selected_copy = selected.copy()
            cover_count_copy = cover_count.copy()
            
            # Perform Swap
            selected.remove(c)
            selected.append(c_new)
            
            update_cc(cover_count, c, -1)
            update_cc(cover_count, c_new, 1)
            
            # Remove Redundant
            old_size = len(selected)
            remove_redundant()
            
            if len(selected) < old_size:
                improved = True
            else:
                # Revert
                selected = selected_copy
                cover_count = cover_count_copy
                
        if not improved:
            break
            
    return [all_words[i] for i in selected]
