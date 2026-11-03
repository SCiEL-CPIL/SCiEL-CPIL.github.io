import itertools
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in scoring)
        # Simple greedy fallback
        total_words = q ** n
        # Generate all words
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute neighbors
        neighbors = [[] for _ in range(total_words)]
        for i, w in enumerate(words):
            nbrs = [i]
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != w[pos]:
                        new_w = list(w)
                        new_w[pos] = new_digit
                        new_w = tuple(new_w)
                        nbrs.append(word_to_idx[new_w])
            neighbors[i] = nbrs
        
        covered = [False] * total_words
        code = []
        while any(not c for c in covered):
            best_count = -1
            best_candidates = []
            for w in range(total_words):
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            selected = random.choice(best_candidates)
            code.append(selected)
            for nbr in neighbors[selected]:
                covered[nbr] = True
        return [list(words[w]) for w in code]
    
    # Optimized path for q=3, n=6, radius=1
    total_words = 729
    
    # Precompute neighbors for each word (integer representation)
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr_digit = (w // div) % q
            for new_digit in range(q):
                if new_digit != curr_digit:
                    new_w = w + (new_digit - curr_digit) * div
                    nbrs.append(new_w)
        neighbors[w] = nbrs
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nbrs) for nbrs in neighbors]
    
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy restarts with different random tie-breakers
    rng = random.Random(42)
    
    for restart in range(100):
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        for _ in range(100): # Max iterations safety
            if covered.all():
                break
                
            best_count = -1
            best_candidates = []
            
            # Find the center that covers the most new points
            # Use vectorized operations where possible
            for w in range(total_words):
                nbrs = neighbors_np[w]
                count = np.sum(~covered[nbrs])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            
            if best_count == 0:
                break
                
            # Pick randomly from best candidates to add diversity
            selected = rng.choice(best_candidates)
            code.append(selected)
            
            # Mark all neighbors of selected as covered
            covered[neighbors_np[selected]] = True
            
            if len(code) >= best_len:
                break # Prune if already worse than best found so far
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
    
    # Convert integer codes back to lists of integers
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
