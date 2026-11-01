import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    total_words = q ** n
    code = []
    code_set = set()
    cover_count = np.zeros(total_words, dtype=int)
    
    # Precompute neighbors
    digits = []
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        digits.append(d)
        
    neighbors = []
    for i in range(total_words):
        d_i = digits[i]
        neigh = [i]
        for pos in range(n):
            for new_digit in range(q):
                if new_digit == d_i[pos]:
                    continue
                diff = (new_digit - d_i[pos]) * (q ** pos)
                neigh.append(i + diff)
        neighbors.append(neigh)
        
    # Precompute "who covers me" (reverse neighbors)
    # Actually, neighbors is symmetric for radius 1.
    # So neighbors[cw] is the list of words that cover cw.
    
    # Greedy Phase
    while time.time() < deadline:
        if np.min(cover_count) >= 1:
            break
            
        best_word = -1
        best_score = -1
        
        for c in range(total_words):
            if c in code_set:
                continue
            score = 0
            for nb in neighbors[c]:
                if cover_count[nb] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_word = c
                
        if best_word == -1:
            break
            
        code.append(best_word)
        code_set.add(best_word)
        for nb in neighbors[best_word]:
            cover_count[nb] += 1
            
    # Local Search Phase
    while time.time() < deadline:
        removed = False
        
        for c in code:
            removable = True
            for nb in neighbors[c]:
                if cover_count[nb] < 2:
                    removable = False
                    break
            if removable:
                code.remove(c)
                code_set.remove(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                removed = True
                break
        
        if removed:
            continue
            
        if not code:
            break
            
        swapped = False
        for _ in range(20):
            if time.time() > deadline:
                break
                
            c_old = random.choice(code)
            critical_words = [nb for nb in neighbors[c_old] if cover_count[nb] == 1]
            
            if not critical_words:
                continue
                
            # Find candidates that cover all critical_words
            # Intersection of neighbors of critical_words
            candidates = set(neighbors[critical_words[0]])
            for cw in critical_words[1:]:
                candidates &= set(neighbors[cw])
                
            # Filter candidates: must not be in code
            valid_candidates = [c for c in candidates if c not in code_set]
            
            if not valid_candidates:
                continue
                
            # Score valid candidates
            best_new = -1
            best_score = -1
            
            for c_new in valid_candidates:
                score = 0
                for nb in neighbors[c_new]:
                    if cover_count[nb] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_new = c_new
                    
            if best_new != -1:
                code.remove(c_old)
                code_set.remove(c_old)
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                    
                code.append(best_new)
                code_set.add(best_new)
                for nb in neighbors[best_new]:
                    cover_count[nb] += 1
                swapped = True
                break
                
        if not swapped:
            # Perturbation
            # Add a random word
            available = [w for w in range(total_words) if w not in code_set]
            if available:
                c_rand = random.choice(available)
                code.append(c_rand)
                code_set.add(c_rand)
                for nb in neighbors[c_rand]:
                    cover_count[nb] += 1

    result = []
    for c in code:
        word = []
        x = c
        for _ in range(n):
            word.append(x % q)
            x //= q
        result.append(word)
        
    return result
