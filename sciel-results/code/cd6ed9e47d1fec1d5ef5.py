import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute words and neighbors
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for u in words:
        u_list = list(u)
        neigh = [word_to_idx[tuple(u_list)]]
        for j in range(n):
            old_val = u_list[j]
            for v in range(q):
                if v != old_val:
                    u_list[j] = v
                    neigh.append(word_to_idx[tuple(u_list)])
            u_list[j] = old_val
        neighbors.append(neigh)
        
    def greedy_max_coverage():
        uncovered = [True] * N
        code = []
        cover_count = [0] * N
        uncovered_count = N
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            best_c = -1
            max_new = 0
            
            # Find best codeword
            for c in range(N):
                new = 0
                for w in neighbors[c]:
                    if uncovered[w]:
                        new += 1
                        if new > max_new: # Early exit if possible? No, need exact max for this c
                            pass 
                # Optimization: if new == 0, skip.
                if new > max_new:
                    max_new = new
                    best_c = c
                    
            if best_c == -1:
                break
                
            code.append(best_c)
            for w in neighbors[best_c]:
                if uncovered[w]:
                    uncovered[w] = False
                    uncovered_count -= 1
                cover_count[w] += 1
                
        return code, cover_count

    def local_search(code, cover_count):
        # code is list of indices
        # cover_count is list of ints
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Randomize order of code indices to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                c = code[i]
                
                # Check if c is redundant
                redundant = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove c
                    # Swap with last element to remove efficiently
                    last = code[-1]
                    code[i] = last
                    code.pop()
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    improved = True
                    # Restart loop to check new redundancies
                    break 
                    
            if improved:
                continue
                
            # Try swaps
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                c = code[i]
                
                # Try swapping c with a neighbor w
                # Pick a random neighbor to try
                # To be more effective, try neighbors that cover "critical" words
                # For now, try a few random neighbors
                tried = set()
                for _ in range(5): # Try 5 random neighbors
                    w = random.choice(neighbors[c])
                    if w in tried:
                        continue
                    tried.add(w)
                    
                    # Check if swap is valid
                    # Decrement c, Increment w
                    # Check min count
                    valid = True
                    min_c = float('inf')
                    for x in neighbors[c]:
                        cnt = cover_count[x] - 1
                        if cnt == 0:
                            valid = False
                            break
                        if cnt < min_c: min_c = cnt
                    if valid:
                        for x in neighbors[w]:
                            cnt = cover_count[x] + 1
                            # No upper limit check needed for validity, just >0 for c's part
                            # But we need to ensure w's neighbors don't go negative? No, we are adding.
                            # We only care if c's neighbors drop to 0.
                            pass
                    
                    if valid:
                        # Perform swap
                        for x in neighbors[c]:
                            cover_count[x] -= 1
                        for x in neighbors[w]:
                            cover_count[x] += 1
                        code[i] = w
                        improved = True
                        break # Restart loop after swap to allow removals
                
                if improved:
                    break
                    
        return code, cover_count

    best_code = []
    best_size = float('inf')
    
    # Run multiple strategies
    strategies = [
        ("greedy_max", lambda: greedy_max_coverage()),
        # Could add randomized greedy here
    ]
    
    # Execute first strategy
    code, counts = greedy_max_coverage()
    code, counts = local_search(code, counts)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code[:]
        
    # If time remains, try a few random restarts with randomized greedy
    # Randomized Greedy: Pick random uncovered word, add best coverer for it.
    def greedy_randomized():
        uncovered = [True] * N
        code = []
        cover_count = [0] * N
        uncovered_count = N
        uncovered_indices = list(range(N))
        
        while uncovered_count > 0 and time.time() < deadline:
            # Pick random uncovered word
            # To avoid O(N) search for random uncovered, we can maintain a list
            # But N is small, so random.choice(uncovered_indices) and filter is ok?
            # Better: maintain a list of uncovered indices
            if not uncovered_indices:
                break
            target = random.choice(uncovered_indices)
            if not uncovered[target]:
                # It was covered by previous steps? 
                # We remove from uncovered_indices when we cover it.
                # So target is always uncovered.
                pass
                
            best_c = -1
            max_new = 0
            for c in range(N):
                new = 0
                for w in neighbors[c]:
                    if uncovered[w]:
                        new += 1
                if new > max_new:
                    max_new = new
                    best_c = c
                    
            if best_c == -1:
                break
                
            code.append(best_c)
            # Update uncovered and counts
            new_uncovered_indices = []
            for w in neighbors[best_c]:
                if uncovered[w]:
                    uncovered[w] = False
                    uncovered_count -= 1
                    # Don't add to new_uncovered_indices
                else:
                    # It was already covered, keep in list? No, we need to remove it from list
                    pass
                cover_count[w] += 1
            
            # Rebuild uncovered_indices list is O(N^2) in worst case.
            # Better: just iterate all N to check uncovered status?
            # Since N=2187, and steps ~200, 200*2187 = 400k checks. Fine.
            uncovered_indices = [i for i in range(N) if uncovered[i]]
            
        return code, cover_count

    # Run randomized greedy a few times
    for i in range(10):
        if time.time() > deadline - 20:
            break
        code, counts = greedy_randomized()
        code, counts = local_search(code, counts)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert indices back to words
    return [list(words[c]) for c in best_code]
