import time
import random
import numpy as np
import itertools

_NEIGHBOR_CACHE = {}

def _get_neighbors(q, n):
    key = (q, n)
    if key not in _NEIGHBOR_CACHE:
        total = q ** n
        neighbors = [[] for _ in range(total)]
        
        def to_digits(x, n, q):
            d = [0] * n
            for i in range(n):
                d[i] = x % q
                x //= q
            return d
        
        def from_digits(d, q):
            x = 0
            for i in range(len(d)-1, -1, -1):
                x = x * q + d[i]
            return x

        for x in range(total):
            digits = to_digits(x, n, q)
            neighbors[x].append(x)
            for i in range(n):
                orig = digits[i]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[i] = val
                    neighbors[x].append(from_digits(new_digits, q))
        _NEIGHBOR_CACHE[key] = neighbors
    return _NEIGHBOR_CACHE[key]

def _greedy_cover(q, n, neighbors, seed):
    random.seed(seed)
    total_words = q ** n
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        max_covered = -1
        
        for u in uncovered:
            count = 0
            for v in neighbors[u]:
                if v in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_word = u
                if count == len(neighbors[u]):
                    break
        
        if best_word == -1:
            break
            
        codewords.append(best_word)
        
        to_remove = [v for v in neighbors[best_word] if v in uncovered]
        for v in to_remove:
            uncovered.remove(v)
            
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if radius != 1:
        return [[0]*n]
        
    neighbors = _get_neighbors(q, n)
    
    # Run greedy
    best_codewords = _greedy_cover(q, n, neighbors, 42)
    
    def _greedy_cover_randomized(q, n, neighbors, seed):
        random.seed(seed)
        total_words = q ** n
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            best_word = -1
            max_covered = -1
            
            u_list = list(uncovered)
            random.shuffle(u_list)
            
            for u in u_list:
                count = 0
                for v in neighbors[u]:
                    if v in uncovered:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_word = u
                    if count == len(neighbors[u]):
                        break
            
            if best_word == -1:
                break
                
            codewords.append(best_word)
            
            to_remove = [v for v in neighbors[best_word] if v in uncovered]
            for v in to_remove:
                uncovered.remove(v)
                
        return codewords

    best_size = len(best_codewords)
    
    seeds = [1, 2, 3, 4, 5]
    for s in seeds:
        if time.time() > deadline - 10:
            break
        cw = _greedy_cover_randomized(q, n, neighbors, s)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            
    def to_list(x, n, q):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d
    
    result = [to_list(c, n, q) for c in best_codewords]
    return result
