import numpy as np
import itertools
import time

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute sphere masks and indices
    sphere_masks = [0] * total_words
    sphere_indices = [[] for _ in range(total_words)]
    
    for i, word in enumerate(all_words):
        mask = 1 << i
        inds = [i]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    j = word_to_idx[new_word]
                    mask |= (1 << j)
                    inds.append(j)
        sphere_masks[i] = mask
        sphere_indices[i] = inds
        
    # Helper for bit counting
    if hasattr(0, 'bit_count'):
        def _bit_count(x):
            return x.bit_count()
    else:
        def _bit_count(x):
            return bin(x).count('1')

    def greedy_cover(uncovered_mask):
        new_code = []
        unc = uncovered_mask
        while unc:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                common = sphere_masks[i] & unc
                if common == 0:
                    continue
                count = _bit_count(common)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_count == 0:
                break
            new_code.append(best_idx)
            unc &= ~sphere_masks[best_idx]
        return new_code

    def reduce_code(code_indices):
        if not code_indices:
            return []
        cover_count = np.zeros(total_words, dtype=np.int8)
        for idx in code_indices:
            for w_idx in sphere_indices[idx]:
                cover_count[w_idx] += 1
                
        current_code = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_code)):
                c_idx = current_code[i]
                is_redundant = True
                for w_idx in sphere_indices[c_idx]:
                    if cover_count[w_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w_idx in sphere_indices[c_idx]:
                        cover_count[w_idx] -= 1
                    current_code.pop(i)
                    changed = True
                    break
        return current_code

    start_time = time.time()
    best_code = greedy_cover((1 << total_words) - 1)
    best_code = reduce_code(best_code)
    best_size = len(best_code)
    
    # Iterated Local Search
    # Try to improve by removing one codeword and re-greedy-ing
    # We iterate through all codewords to find the best improvement
    improved = True
    while improved and (time.time() - start_time) < 100:
        improved = False
        for i in range(len(best_code)):
            if time.time() - start_time > 100:
                break
            rem_c = best_code[i]
            # Compute covered mask without rem_c
            covered_mask = 0
            for j, c in enumerate(best_code):
                if j != i:
                    covered_mask |= sphere_masks[c]
                    
            unc_mask = ((1 << total_words) - 1) & ~covered_mask
            
            if unc_mask == 0:
                new_code = [c for j, c in enumerate(best_code) if j != i]
                new_code = reduce_code(new_code)
                if len(new_code) < best_size:
                    best_code = new_code
                    best_size = len(best_code)
                    improved = True
                    break # Restart with new code
                else:
                    continue
            
            added = greedy_cover(unc_mask)
            new_code = [c for j, c in enumerate(best_code) if j != i]
            new_code.extend(added)
            new_code = reduce_code(new_code)
            
            if len(new_code) < best_size:
                best_code = new_code
                best_size = len(new_code)
                improved = True
                break # Restart with new code
                
    return [list(all_words[idx]) for idx in best_code]
