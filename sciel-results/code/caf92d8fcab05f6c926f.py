import numpy as np
from scipy.sparse import csr_matrix
import itertools
import time
import random

def construct(q, n, radius):
    # Specific optimization for the given task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # Simple greedy
        total_words = q ** n
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = []
        for i in range(total_words):
            w = words[i]
            neigh = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    neigh.append(word_to_idx[tuple(new_w)])
            neighbors.append(neigh)
            
        rows, cols, data = [], [], []
        for i in range(total_words):
            for j in neighbors[i]:
                rows.append(i); cols.append(j); data.append(1)
        A = csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
        
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        while np.sum(uncovered) > 0:
            gains = A.dot(uncovered)
            c = np.argmax(gains)
            code.append(c)
            uncovered[neighbors[c]] = 0
        return [list(words[idx]) for idx in code]

    # Precomputation for q=3, n=7
    total_words = 3 ** 7
    words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(7):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
        
    rows = []
    cols = []
    data = []
    for i in range(total_words):
        for j in neighbors[i]:
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: Direct Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    fallback_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            fallback_code.append(word_to_idx[tuple(h + list(tail))])
            
    best_code = fallback_code
    best_size = len(best_code)
    
    def reduce_code(code_list, seed):
        rng = random.Random(seed)
        cover_count = np.zeros(total_words, dtype=np.int32)
        active_set = set(code_list)
        
        for c in active_set:
            cover_count[neighbors[c]] += 1
            
        for _ in range(100):
            changed = False
            shuffled_code = list(active_set)
            rng.shuffle(shuffled_code)
            
            for c in shuffled_code:
                if c not in active_set:
                    continue
                if np.min(cover_count[neighbors[c]]) > 1:
                    active_set.remove(c)
                    cover_count[neighbors[c]] -= 1
                    changed = True
            if not changed:
                break
        return list(active_set)

    restart_count = 0
    while time.time() < deadline:
        # Greedy Construction
        uncovered = np.ones(total_words, dtype=np.int8)
        greedy_code = []
        while np.sum(uncovered) > 0:
            gains = A.dot(uncovered)
            c = np.argmax(gains)
            greedy_code.append(c)
            uncovered[neighbors[c]] = 0
            
        # Reduction on Greedy
        for seed in range(5):
            reduced = reduce_code(greedy_code, seed + restart_count)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_code = reduced
                
        # Occasionally reduce Fallback
        if restart_count % 20 == 0:
            for seed in range(2):
                reduced = reduce_code(fallback_code, seed + restart_count + 1000)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_code = reduced

        restart_count += 1
        
    return [list(words[idx]) for idx in best_code]
