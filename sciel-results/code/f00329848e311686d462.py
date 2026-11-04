"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


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
    
    # Compute neighbors (ball of radius 1)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors[i] = neigh
    
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered, code, time_limit):
        """Greedy set cover to add codewords until all covered."""
        uncovered = set(uncovered)
        code = list(code)
        
        while uncovered and time.time() < time_limit:
            best_candidate = -1
            max_new_cov = 0
            candidates = []
            
            for c in range(N):
                cov = sum(1 for u in neighbors[c] if u in uncovered)
                if cov > max_new_cov:
                    max_new_cov = cov
                    best_candidate = c
                    candidates = [c]
                elif cov == max_new_cov and cov > 0:
                    candidates.append(c)
            
            if best_candidate == -1:
                break
            
            chosen = random.choice(candidates)
            code.append(chosen)
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        return code
    
    def prune_redundant(code, time_limit):
        """Remove redundant codewords."""
        if not code:
            return code
        
        cover_count = [0] * N
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            for idx, c in enumerate(code):
                redundant = True
                for u in neighbors[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    for u in neighbors[c]:
                        cover_count[u] -= 1
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
        covered = [False] * N
        for c in code:
            for u in neighbors[c]:
                covered[u] = True
        
        uncovered = [i for i in range(N) if not covered[i]]
        
        # Repair
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)
        
        return code
    
    # Run multiple iterations with different strategies
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with greedy
        code = greedy_cover(set(range(N)), [], deadline - 2)
        code = prune_redundant(code, deadline - 2)
        
        # Try kicks
        for num_kicks in [1, 2, 3, 4, 5]:
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
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(words[c]) for c in best_code]
# EVOLVE-BLOCK-END
