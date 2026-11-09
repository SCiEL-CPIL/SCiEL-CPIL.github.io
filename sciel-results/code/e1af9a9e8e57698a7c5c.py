import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Total space size
    space_size = q ** n
    
    # Precompute neighbors for all words in the space
    # Map word to index
    # Word is a tuple of n integers
    # Index is base-q integer
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    def index_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
        
    # Precompute neighbors
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(space_size)]
    
    # Generate all words and their neighbors
    # To speed up, we can generate neighbors on the fly or precompute.
    # Precomputation:
    for idx in range(space_size):
        w = index_to_word(idx)
        # Self
        neighbors[idx].append(idx)
        # Distance 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = list(w)
                new_w[pos] = val
                new_idx = word_to_index(tuple(new_w))
                neighbors[idx].append(new_idx)
                
    # Greedy Set Cover
    # uncovered: boolean array
    # count: number of uncovered neighbors for each word
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations with randomization
    # The greedy choice is deterministic if we always pick the first max.
    # We can break ties randomly.
    
    for run in range(20): # Try a few runs
        if time.time() > deadline:
            break
            
        uncovered = [True] * space_size
        count = [0] * space_size
        
        # Initialize counts
        for i in range(space_size):
            # All neighbors are uncovered initially
            count[i] = len(neighbors[i])
            
        code = []
        
        # To pick max count efficiently, we can just scan.
        # Space size 6561 is small.
        
        while True:
            # Check if all covered
            # We can maintain a counter of uncovered words
            # Or just check if max count is 0?
            # If max count is 0, it means no word has any uncovered neighbors.
            # This implies all words are covered?
            # If a word u is uncovered, then for any v, if u is in N(v), count[v] >= 1.
            # Since u is in N(u), count[u] >= 1.
            # So if max(count) == 0, all are covered.
            
            max_count = -1
            best_idx = -1
            candidates = []
            
            for i in range(space_size):
                if count[i] > max_count:
                    max_count = count[i]
                    best_idx = i
                    candidates = [i]
                elif count[i] == max_count:
                    candidates.append(i)
                    
            if max_count <= 0:
                break
                
            # Pick one candidate randomly
            c_star = random.choice(candidates)
            code.append(index_to_word(c_star))
            
            # Update uncovered and counts
            for u in neighbors[c_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    # Update counts for all neighbors of u
                    for v in neighbors[u]:
                        count[v] -= 1
                        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert list of tuples to list of lists
    if best_code is None:
        # Fallback to direct sum if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(w) for w in best_code]
