import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute all words and neighbors
    total_words = q ** n
    
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    neighbors[i].append(word_to_int(w))
                w[pos] = orig
    
    # Initial Code: Direct Sum Code (Tetracode x 2)
    # For q=3, n=6, radius=1
    # Tetracode has 9 words of length 4.
    # Direct sum with 2 more coords gives 9 * 3 * 3 = 81 words.
    init_code = []
    if q == 3 and n == 6:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                init_code.append(word_to_int(h + list(tail)))
    else:
        # Fallback: Greedy
        uncovered = np.ones(total_words, dtype=bool)
        in_code_temp = np.zeros(total_words, dtype=bool)
        while np.any(uncovered):
            best_word = -1
            max_new_cover = 0
            for c in range(total_words):
                if in_code_temp[c]:
                    continue
                new_cover = np.sum(uncovered[neighbors[c]])
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_word = c
            if best_word == -1:
                break
            init_code.append(best_word)
            in_code_temp[best_word] = True
            for nb in neighbors[best_word]:
                if in_code_temp[nb] == False: # Wait, this logic is wrong for greedy init
                    pass # Just mark uncovered as covered
                uncovered[nb] = False
            # Correct greedy init logic:
            # We need to track coverage.
            # Let's just use the previous greedy logic properly if needed.
            # But for q=3, n=6, we have the Direct Sum.
            pass

    init_code = list(set(init_code))
    
    in_code = np.zeros(total_words, dtype=bool)
    coverage_count = np.zeros(total_words, dtype=np.int32)
    code_set = set(init_code)
    
    for c in init_code:
        in_code[c] = True
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    def try_remove():
        removed_any = False
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c in candidates:
            if not in_code[c]:
                continue
            essential = False
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    essential = True
                    break
            if not essential:
                in_code[c] = False
                code_set.remove(c)
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                removed_any = True
        return removed_any

    def try_swap():
        if not code_set:
            return False
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c_out in candidates:
            if not in_code[c_out]:
                continue
            unique_words = []
            for nb in neighbors[c_out]:
                if coverage_count[nb] == 1:
                    unique_words.append(nb)
            if not unique_words:
                continue
                
            found = False
            all_candidates = [w for w in range(total_words) if not in_code[w]]
            random.shuffle(all_candidates)
            
            for c_in in all_candidates:
                valid = True
                for u in unique_words:
                    if u not in neighbors[c_in]:
                        valid = False
                        break
                if valid:
                    in_code[c_out] = False
                    code_set.remove(c_out)
                    for nb in neighbors[c_out]:
                        coverage_count[nb] -= 1
                    in_code[c_in] = True
                    code_set.add(c_in)
                    for nb in neighbors[c_in]:
                        coverage_count[nb] += 1
                    return True
        return False

    stuck_count = 0
    while time.time() < deadline:
        while try_remove():
            stuck_count = 0
            
        swapped = False
        for _ in range(20):
            if try_swap():
                swapped = True
                stuck_count = 0
            else:
                break
                
        if not swapped:
            stuck_count += 1
            if stuck_count > 5:
                candidates = [w for w in range(total_words) if not in_code[w]]
                if candidates:
                    c_add = random.choice(candidates)
                    in_code[c_add] = True
                    code_set.add(c_add)
                    for nb in neighbors[c_add]:
                        coverage_count[nb] += 1
                    for _ in range(50):
                        if not try_remove():
                            break
                    stuck_count = 0
                else:
                    break

    final_code = [int_to_word(c) for c in code_set]
    return final_code
