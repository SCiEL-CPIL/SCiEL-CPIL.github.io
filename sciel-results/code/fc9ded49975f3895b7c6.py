import random
import numpy as np
import time

def construct(q, n, radius):
    total_words = q ** n
    all_words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    # Precompute coverage
    coverage = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nw = tuple(nw)
                idx = 0
                for j in range(n):
                    idx = idx * q + nw[j]
                covered.append(idx)
        coverage.append(covered)
    
    best_solution = None
    best_size = total_words
    start_time = time.time()
    time_limit = 115.0
    
    # Greedy + Local Search
    while time.time() - start_time < time_limit:
        codewords = set()
        uncovered = set(range(total_words))
        
        # Greedy Phase
        while uncovered:
            best_c = -1
            best_count = -1
            best_tie = []
            
            for c in range(total_words):
                if c in codewords:
                    continue
                cnt = 0
                for j in coverage[c]:
                    if j in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    best_tie = [c]
                elif cnt == best_count and cnt > 0:
                    best_tie.append(c)
            
            if best_c == -1:
                break
                
            # Randomize among top candidates to diversify
            if len(best_tie) > 1:
                # Pick from top 5 or so
                k = min(5, len(best_tie))
                chosen = random.choice(best_tie[:k])
            else:
                chosen = best_c
                
            codewords.add(chosen)
            for j in coverage[chosen]:
                if j in uncovered:
                    uncovered.discard(j)
        
        if not codewords:
            continue
            
        # Local Search
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            counts[coverage[c]] += 1
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # 1. Try Removals
            cw_list = list(codewords)
            random.shuffle(cw_list)
            
            for c in cw_list:
                if c not in codewords:
                    continue
                counts[coverage[c]] -= 1
                if np.any(counts == 0):
                    counts[coverage[c]] += 1
                else:
                    codewords.discard(c)
                    improved = True
                    break # Restart
            
            if not codewords:
                break
                
            # 2. Try Swaps if no removal worked
            if not improved:
                # Try a few random swaps
                swaps_attempted = 0
                swaps_max = 20
                while swaps_attempted < swaps_max and (time.time() - start_time < time_limit):
                    swaps_attempted += 1
                    c = random.choice(list(codewords))
                    w = random.randint(0, total_words - 1)
                    if w in codewords:
                        continue
                    
                    # Tentative swap
                    counts[coverage[c]] -= 1
                    counts[coverage[w]] += 1
                    
                    if not np.any(counts == 0):
                        codewords.discard(c)
                        codewords.add(w)
                        improved = True
                        break
                    else:
                        counts[coverage[w]] -= 1
                        counts[coverage[c]] += 1
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_solution = set(codewords)
            
    if best_solution is None:
        return [list(all_words[i]) for i in range(total_words)]
        
    return [list(all_words[i]) for i in best_solution]
