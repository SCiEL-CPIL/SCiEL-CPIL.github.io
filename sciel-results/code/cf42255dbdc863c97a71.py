import time
import itertools
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Generate the seed code
    # Direct sum of tetracode and full space
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # 2. Setup coverage counter
    # Map word to index: w[0]*3^7 + w[1]*3^6 + ... + w[7]*3^0
    # Or simpler: treat word as base-3 number.
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)

    space_size = 3 ** n
    cover_count = [0] * space_size
    
    # Precompute neighbors for all words in space? 
    # space_size is 6561. 6561 * 17 ~ 111k entries. This is fine.
    # But we only need neighbors for codewords initially, and potentially for others if we add/remove.
    # Actually, we only need to update counts for neighbors of the codeword being added/removed.
    
    def get_neighbors(w):
        neighbors = [w]
        for i in range(n):
            orig = w[i]
            for val in range(3):
                if val != orig:
                    new_w = list(w)
                    new_w[i] = val
                    neighbors.append(tuple(new_w))
        return neighbors

    # Initialize coverage
    code_set = set(seed_code)
    code_list = list(seed_code)
    
    for c in code_list:
        idx = word_to_idx(c)
        for nb in get_neighbors(c):
            cover_count[word_to_idx(nb)] += 1
            
    # 3. Redundancy removal loop
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Check each codeword
        # To be efficient, we can iterate through a copy of the list
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            idx = word_to_idx(c)
            # Check if all neighbors are covered by > 1 codeword
            redundant = True
            for nb in get_neighbors(c):
                if cover_count[word_to_idx(nb)] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                for nb in get_neighbors(c):
                    cover_count[word_to_idx(nb)] -= 1
                code_list.pop(i)
                improved = True
                
    # 4. Return the result
    # Convert back to lists of ints as required
    return [list(c) for c in code_list]
