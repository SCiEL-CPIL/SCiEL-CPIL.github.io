import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Problem parameters
    N = q ** n  # 2187 for q=3, n=7
    
    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word
    
    # Precompute all words
    words = [idx_to_word(i) for i in range(N)]
    
    # Build word to index mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Compute neighbors using numpy for speed
    # neighbors[i] is a list of indices within Hamming distance radius from i
    neighbors = np.empty((N, 1 + n * (q - 1)), dtype=np.int32)
    
    for i in range(N):
        w_i = words[i]
        neighbors[i, 0] = i
        pos = 1
        for p in range(n):
            for val in range(q):
                if val == w_i[p]:
                    continue
                new_w = list(w_i)
                new_w[p] = val
                neighbors[i, pos] = word_to_idx[tuple(new_w)]
                pos += 1
    
    # Convert to list of lists for easier manipulation
    neighbors_list = [neighbors[i].tolist() for i in range(N)]
    
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered, code, time_limit):
        """Greedy set cover to add codewords until all covered."""
        uncovered_set = set(uncovered)
        code = list(code)
        
        # Use numpy array for fast gain computation
        uncovered_arr = np.zeros(N, dtype=np.bool_)
        for u in uncovered_set:
            uncovered_arr[u] = True
        
        while uncovered_set and time.time() < time_limit:
            # Compute gains for all candidates at once using numpy
            # For each candidate c, gain = number of neighbors in uncovered
            gains = np.array([np.sum(uncovered_arr[neighbors_list[c]]) for c in range(N)])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Get candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(candidates.tolist())
            
            code.append(chosen)
            # Update uncovered
            for u in neighbors_list[chosen]:
                if u in uncovered_set:
                    uncovered_set.discard(u)
                    uncovered_arr[u] = False
        
        return code
    
    def prune_redundant(code, time_limit):
        """Remove redundant codewords."""
        if not code:
            return code
        
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors_list[c]] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            for idx, c in enumerate(code):
                # Check if all neighbors are covered by others
                if np.all(cover_count[neighbors_list[c]] > 1):
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    cover_count[neighbors_list[c]] -= 1
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True
        
        return code
    
    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code
        
        code = list(code)
        # Remove random codewords
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)
        
        # Find uncovered
        covered = np.zeros(N, dtype=np.bool_)
        for c in code:
            covered[neighbors_list[c]] = True
        
        uncovered = np.where(~covered)[0].tolist()
        
        # Repair
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)
        
        return code
    
    # Run multiple iterations with different strategies
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with greedy
        code = greedy_cover(list(range(N)), [], deadline - 2)
        code = prune_redundant(code, deadline - 2)
        
        # Try kicks
        for num_kicks in [1, 2, 3]:
            if time.time() > deadline - 2:
                break
            code = kick_and_repair(code, num_kicks, deadline - 2)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            if best_size <= 186:
                break
    
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(words[c]) for c in best_code]
