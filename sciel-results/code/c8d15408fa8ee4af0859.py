import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Precompute neighbors
    def int_to_base_q(x, q, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def base_q_to_int(digits, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + digits[i]
        return x

    neighbors = []
    for i in range(total_words):
        digs = int_to_base_q(i, q, n)
        neigh = [i]
        for pos in range(n):
            original = digs[pos]
            for val in range(q):
                if val == original:
                    continue
                new_digs = digs.copy()
                new_digs[pos] = val
                new_int = base_q_to_int(new_digs, q)
                neigh.append(new_int)
        neighbors.append(neigh)
        
    def greedy_construct():
        cover_count = np.zeros(total_words, dtype=np.int8)
        uncovered = total_words
        code = []
        
        while uncovered > 0:
            best_c = -1
            best_gain = -1
            # Iterate all candidates
            for c in range(total_words):
                gain = 0
                for w in neighbors[c]:
                    if cover_count[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    if best_gain == len(neighbors[c]):
                        # Can't get better than covering all its neighbors
                        # But another c might also have this gain. 
                        # Breaking here is safe for "some" optimal step, 
                        # but might miss a "better" optimal step? 
                        # No, gain is max possible for this c. 
                        # If we break, we pick the first one with max gain.
                        # This is standard greedy.
                        break
            
            if best_c == -1:
                break
                
            code.append(best_c)
            for w in neighbors[best_c]:
                if cover_count[w] == 0:
                    uncovered -= 1
                cover_count[w] += 1
                
        return code, cover_count

    def prune(code, cover_count):
        uncovered = np.sum(cover_count == 0)
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle pop safely, or just use indices
            # Randomize order to break symmetries?
            idxs = list(range(len(code)))
            random.shuffle(idxs)
            for idx in idxs:
                c = code[idx]
                # Tentatively remove
                for w in neighbors[c]:
                    cover_count[w] -= 1
                    if cover_count[w] == 0:
                        uncovered += 1
                if uncovered == 0:
                    code.pop(idx)
                    changed = True
                else:
                    for w in neighbors[c]:
                        cover_count[w] += 1
                        if cover_count[w] == 1:
                            uncovered -= 1
        return code, cover_count

    best_size = float('inf')
    best_code = None
    
    # Run multiple restarts
    while time.time() < deadline - 10:
        code, cover_count = greedy_construct()
        code, cover_count = prune(code, cover_count)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for c in best_code:
        digs = int_to_base_q(c, q, n)
        result.append(digs)
        
    return result
