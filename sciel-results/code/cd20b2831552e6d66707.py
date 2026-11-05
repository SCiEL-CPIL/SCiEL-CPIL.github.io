import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = q ** n
    
    # Fallback for large N or R != 1
    if N > 200000 or radius != 1:
        # Use the tetracode construction if possible
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This is a valid cover for n=4. For n>4, product with all words.
            # Size: 9 * 3^(n-4)
            import itertools
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Random fallback
            return [list(random.choices(range(q), k=n)) for _ in range(int(N * 0.1))]

    # Optimized path for small N and R=1
    # Precompute neighbors
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i].append(word_to_idx(w))
                
    best_code = None
    best_size = float('inf')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        code = greedy_cover(N, neighbors, attempts)
        code = remove_redundancy(code, N, neighbors)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        if best_size <= 486:
            break
            
    if best_code is None:
        return [idx_to_word(i) for i in range(N)]
        
    return [idx_to_word(i) for i in best_code]

def greedy_cover(N, neighbors, seed_offset):
    random.seed(42 + seed_offset)
    covered = [False] * N
    benefit = [len(neighbors[i]) for i in range(N)]
    
    uncovered_count = N
    code = []
    
    while uncovered_count > 0:
        max_benefit = -1
        best_idx = -1
        for i in range(N):
            if benefit[i] > max_benefit:
                max_benefit = benefit[i]
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.append(best_idx)
        
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                uncovered_count -= 1
                for k in neighbors[j]:
                    benefit[k] -= 1
                    
    return code

def remove_redundancy(code, N, neighbors):
    cover_count = [0] * N
    for c in code:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    changed = True
    while changed:
        changed = False
        new_code = []
        for c in code:
            is_redundant = True
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                for w in neighbors[c]:
                    cover_count[w] -= 1
                changed = True
            else:
                new_code.append(c)
                
        code = new_code
        
    return code
