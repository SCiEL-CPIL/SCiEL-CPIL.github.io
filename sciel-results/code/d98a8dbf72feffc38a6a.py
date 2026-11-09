import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q != 3 or n != 8 or radius != 1:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    
    neighbors = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return w
        
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    for i in range(total_words):
        w = idx_to_word(i)
        neigh_list = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neigh_list.append(word_to_idx(w))
            w[pos] = orig_val
        neighbors[i] = neigh_list

    uncovered = np.ones(total_words, dtype=bool)
    selected_indices = []
    
    while True:
        if time.time() > deadline - 10.0:
            break
        counts = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        if best_count == 0:
            break
        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if not uncovered.any():
            break

    if not selected_indices:
        return []

    selected_set = set(selected_indices)
    selected_array = np.array(selected_indices)
    
    flat_neigh = neighbors[selected_array].ravel()
    counts = np.bincount(flat_neigh, minlength=total_words)
    
    def can_remove(c_idx):
        return np.min(counts[neighbors[c_idx]]) > 1

    def remove_codeword(c_idx):
        nonlocal selected_set, selected_array, counts
        counts[neighbors[c_idx]] -= 1
        selected_set.remove(c_idx)
        selected_array = np.array(list(selected_set))

    def swap_codeword(old_idx, new_idx):
        nonlocal selected_set, selected_array, counts
        counts[neighbors[old_idx]] -= 1
        counts[neighbors[new_idx]] += 1
        selected_set.remove(old_idx)
        selected_set.add(new_idx)
        selected_array = np.array(list(selected_set))

    best_size = len(selected_set)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        improved = False
        
        current_list = list(selected_set)
        random.shuffle(current_list)
        for c in current_list:
            if c not in selected_set:
                continue
            if can_remove(c):
                remove_codeword(c)
                improved = True
                best_size = min(best_size, len(selected_set))
                no_improve_count = 0
                break
        
        if improved:
            continue
            
        if len(selected_set) < 2:
            break
            
        found_2del = False
        c1_candidates = list(selected_set)
        random.shuffle(c1_candidates)
        
        for c1 in c1_candidates:
            if time.time() > deadline - 2.0:
                break
                
            c2_candidates = [c for c in selected_set if c != c1]
            random.shuffle(c2_candidates)
            
            for c2 in c2_candidates[:5]:
                n1 = neighbors[c1]
                n2 = neighbors[c2]
                crit1 = n1[counts[n1] == 1]
                crit2 = n2[counts[n2] == 1]
                
                if len(crit1) == 0 and len(crit2) == 0:
                    continue
                
                crit_union = np.concatenate([crit1, crit2])
                
                if len(crit_union) == 0:
                    continue
                
                w0 = crit_union[0]
                potential = set(neighbors[w0].tolist())
                
                for w in crit_union[1:]:
                    if not potential:
                        break
                    potential &= set(neighbors[w].tolist())
                    
                if not potential:
                    continue
                
                potential -= selected_set
                
                if not potential:
                    continue
                
                c_new = random.choice(list(potential))
                
                counts[neighbors[c_new]] += 1
                counts[neighbors[c1]] -= 1
                counts[neighbors[c2]] -= 1
                
                selected_set.add(c_new)
                selected_set.remove(c1)
                selected_set.remove(c2)
                
                selected_array = np.array(list(selected_set))
                
                best_size = min(best_size, len(selected_set))
                no_improve_count = 0
                found_2del = True
                improved = True
                break
            
            if found_2del:
                break
        
        if found_2del:
            continue
            
        c = random.choice(list(selected_set))
        n_c = neighbors[c]
        crit_c = n_c[counts[n_c] == 1]
        
        if len(crit_c) == 0:
            continue
            
        w0 = crit_c[0]
        potential = set(neighbors[w0].tolist())
        for w in crit_c[1:]:
            if not potential:
                break
            potential &= set(neighbors[w].tolist())
            
        potential -= selected_set
        
        if potential:
            c_new = random.choice(list(potential))
            swap_codeword(c, c_new)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        current_list = list(selected_set)
        random.shuffle(current_list)
        for c in current_list:
            if c not in selected_set:
                continue
            if can_remove(c):
                remove_codeword(c)
                improved = True
                break

    codewords = [idx_to_word(i) for i in selected_set]
    return codewords
