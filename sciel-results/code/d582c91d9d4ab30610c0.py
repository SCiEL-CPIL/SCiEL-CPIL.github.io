import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Precompute digits for all 6561 words
    all_words = np.arange(6561, dtype=np.int32)
    digits = np.zeros((6561, 8), dtype=np.int8)
    for j in range(8):
        divisor = 3 ** (7 - j)
        digits[:, j] = (all_words // divisor) % 3
    
    # Precompute neighbor offsets for each (position, digit)
    multipliers = [3 ** (7 - j) for j in range(8)]
    neighbor_offsets = {}
    for j in range(8):
        mult = multipliers[j]
        for d in range(3):
            off1 = ((d + 1) % 3 - d) * mult
            off2 = ((d + 2) % 3 - d) * mult
            neighbor_offsets[(j, d)] = (off1, off2)
    
    # Precompute balls for all words (17 words each)
    # This is 6561 * 17 = ~111k entries, manageable
    balls = np.zeros((6561, 17), dtype=np.int32)
    for w in range(6561):
        balls[w, 0] = w
        for j in range(8):
            d = digits[w, j]
            off1, off2 = neighbor_offsets[(j, d)]
            balls[w, 2*j+1] = w + off1
            balls[w, 2*j+2] = w + off2
    
    def word_to_int(word):
        result = 0
        for d in word:
            result = result * 3 + d
        return result
    
    def int_to_word(w):
        word = []
        for _ in range(8):
            word.append(w % 3)
            w //= 3
        word.reverse()
        return word
    
    # Greedy set cover: repeatedly pick codeword covering most uncovered points
    def greedy_set_cover():
        covered = np.zeros(6561, dtype=bool)
        cw = []
        
        # Track uncovered count
        uncovered_count = 6561
        
        while uncovered_count > 0:
            if time.time() > deadline - 15:
                break
            
            # Find the word that covers the most uncovered points
            best_w = -1
            best_count = -1
            
            # Sample a subset of candidates for speed
            candidates = np.random.choice(6561, size=min(200, 6561), replace=False)
            for w in candidates:
                if covered[w]:
                    continue
                count = np.sum(~covered[balls[w]])
                if count > best_count:
                    best_count = count
                    best_w = w
            
            if best_w == -1 or best_count <= 0:
                # Fallback: pick any uncovered word
                uncovered_idx = np.where(~covered)[0]
                if len(uncovered_idx) == 0:
                    break
                best_w = uncovered_idx[0]
            
            cw.append(best_w)
            for w in balls[best_w]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        return cw, covered
    
    # Try to remove redundant codewords
    def remove_redundant(cw, covered):
        if not cw:
            return cw, covered
        
        # Build coverage count
        cov_count = np.zeros(6561, dtype=np.int16)
        for w in cw:
            for nw in balls[w]:
                cov_count[nw] += 1
        
        # Find removable codewords
        removed = 0
        new_cw = list(cw)
        
        for _ in range(len(new_cw)):
            if time.time() > deadline - 10:
                break
            progress = False
            for i in range(len(new_cw)):
                w = new_cw[i]
                # Check if all words in ball are covered by others
                if all(cov_count[nw] > 1 for nw in balls[w]):
                    for nw in balls[w]:
                        cov_count[nw] -= 1
                    new_cw.pop(i)
                    removed += 1
                    progress = True
                    break
            if not progress:
                break
        
        return new_cw, removed
    
    # Main algorithm: multiple strategies
    best_cw = None
    best_size = float('inf')
    
    # Strategy 1: Tetracode starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    cw_list = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            cw_list.append(h + list(tail))
    cw_ints = [word_to_int(cw) for cw in cw_list]
    
    # Remove redundancies
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        cw_ints, removed = remove_redundant(cw_ints, None)
        if removed == 0:
            break
    
    if len(cw_ints) < best_size:
        best_size = len(cw_ints)
        best_cw = cw_ints
    
    # Strategy 2: Randomized greedy set cover with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        cw, covered = greedy_set_cover()
        
        if not np.all(covered):
            continue
        
        # Remove redundancies
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            cw, removed = remove_redundant(cw, covered)
            if removed == 0:
                break
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_cw = cw
    
    return [int_to_word(w) for w in best_cw]
