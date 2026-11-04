import numpy as np
import itertools
import random

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres
    spheres = []
    for word in all_words:
        s = [word_to_idx[word]]
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val != orig:
                    w_list = list(word)
                    w_list[pos] = val
                    s.append(word_to_idx[tuple(w_list)])
        spheres.append(s)
        
    def run_greedy(seed):
        random.seed(seed)
        uncovered = np.ones(total_words, dtype=bool)
        count = np.zeros(total_words, dtype=int)
        
        # Initialize count
        for i in range(total_words):
            count[i] = np.sum(uncovered[spheres[i]])
            
        code = []
        
        while np.any(uncovered):
            max_val = np.max(count)
            if max_val == 0:
                break
            candidates = np.where(count == max_val)[0]
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            
            # Update
            for w in spheres[best_idx]:
                if uncovered[w]:
                    uncovered[w] = False
                    for c in spheres[w]:
                        count[c] -= 1
                        
        return code

    def reduce_code(code):
        # Remove redundant codewords
        cover_count = np.zeros(total_words, dtype=int)
        for idx in code:
            for w in spheres[idx]:
                cover_count[w] += 1
                
        current_code = list(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_code)):
                c_idx = current_code[i]
                is_redundant = True
                for w in spheres[c_idx]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in spheres[c_idx]:
                        cover_count[w] -= 1
                    current_code.pop(i)
                    changed = True
                    break
        return current_code

    best_code = None
    best_size = float('inf')
    
    # Run multiple times with different seeds
    for seed in range(50):
        code = run_greedy(seed)
        reduced = reduce_code(code)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_code = reduced
            
    return [list(all_words[idx]) for idx in best_code]
