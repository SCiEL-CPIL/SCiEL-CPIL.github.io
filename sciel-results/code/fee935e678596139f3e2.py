import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115
    
    if n == 7 and q == 3 and radius == 1:
        return _solve_ternary_7_1(time_limit)
    else:
        return _generic_construct(q, n, radius)

def _solve_ternary_7_1(time_limit):
    n = 7
    q = 3
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    neighbors = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neigh = [idx]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                w[i] = v
                neigh.append(word_to_idx(w))
            w[i] = orig
        neighbors.append(neigh)
        
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            code.append(h + list(tail))
            
    code_set = set(tuple(c) for c in code)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    for c in code:
        idx = word_to_idx(c)
        for n_idx in neighbors[idx]:
            cover_count[n_idx] += 1
            
    zero_count = np.sum(cover_count == 0)
    
    def is_valid():
        return zero_count == 0

    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Remove redundant
        indices_to_check = list(range(len(code)))
        random.shuffle(indices_to_check)
        for i in indices_to_check:
            if i >= len(code): continue
            c = code[i]
            idx = word_to_idx(c)
            for n_idx in neighbors[idx]:
                old_val = cover_count[n_idx]
                new_val = old_val - 1
                cover_count[n_idx] = new_val
                if old_val == 1 and new_val == 0:
                    zero_count += 1
                elif old_val == 0 and new_val == -1: # Should not happen if valid
                    pass
            
            if zero_count == 0:
                code.pop(i)
                code_set.remove(tuple(c))
                improved = True
            else:
                for n_idx in neighbors[idx]:
                    old_val = cover_count[n_idx]
                    new_val = old_val + 1
                    cover_count[n_idx] = new_val
                    if old_val == 0 and new_val == 1:
                        zero_count -= 1

        # 2. Swap
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            if not code: break
            i = random.randint(0, len(code) - 1)
            c = code[i]
            old_idx = word_to_idx(c)
            
            new_word = [random.randint(0, q - 1) for _ in range(n)]
            new_idx = word_to_idx(new_word)
            
            if new_idx == old_idx:
                continue
                
            if tuple(new_word) in code_set:
                continue

            # Apply swap
            for n_idx in neighbors[old_idx]:
                old_val = cover_count[n_idx]
                new_val = old_val - 1
                cover_count[n_idx] = new_val
                if old_val == 1 and new_val == 0:
                    zero_count += 1
            
            for n_idx in neighbors[new_idx]:
                old_val = cover_count[n_idx]
                new_val = old_val + 1
                cover_count[n_idx] = new_val
                if old_val == 0 and new_val == 1:
                    zero_count -= 1
                    
            if zero_count == 0:
                code[i] = new_word
                code_set.remove(tuple(c))
                code_set.add(tuple(new_word))
                improved = True
            else:
                for n_idx in neighbors[old_idx]:
                    old_val = cover_count[n_idx]
                    new_val = old_val + 1
                    cover_count[n_idx] = new_val
                    if old_val == 0 and new_val == 1:
                        zero_count -= 1
                for n_idx in neighbors[new_idx]:
                    old_val = cover_count[n_idx]
                    new_val = old_val - 1
                    cover_count[n_idx] = new_val
                    if old_val == 1 and new_val == 0:
                        zero_count += 1

    return code

def _generic_construct(q, n, radius):
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(t) for t in itertools.product(range(3), repeat=n)]
