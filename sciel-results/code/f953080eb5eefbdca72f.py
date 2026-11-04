import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Total number of words
    N = q ** n
    
    # Precompute neighbors for all words?
    # Or generate on the fly?
    # Generating on the fly is O(n) per word.
    # Precomputing is O(N * n * q) memory and time.
    # N=6561, n=8, q=3.
    # Neighbors per word: 1 + 8*2 = 17.
    # Total neighbor entries: 6561 * 17 ~ 111k.
    # Precomputation is fast and speeds up the main loops.
    
    # Convert word to index: w[0]*q^(n-1) + ... + w[n-1]
    # Convert index to word: divmod
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute neighbors for each index
    # neighbors[i] is a list of indices j such that d(i, j) <= radius
    # For radius 1, it's i and its 1-neighborhood.
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    neighbors[i].append(word_to_idx(w))
            w[pos] = orig

    # Greedy Construction
    covered = [False] * N
    # gain[i] = number of uncovered neighbors of i
    gain = [0] * N
    for i in range(N):
        gain[i] = len(neighbors[i]) # Initially all uncovered
    
    code = []
    
    while True:
        # Check if all covered
        # We can track a count of uncovered words
        # Let's use a variable `uncovered_count`
        # If uncovered_count == 0, break
        
        # Find max gain
        max_gain = -1
        best_idx = -1
        for i in range(N):
            if gain[i] > max_gain:
                max_gain = gain[i]
                best_idx = i
        
        if max_gain == 0:
            break
            
        # Add best_idx to code
        code.append(best_idx)
        
        # Update covered and gains
        # For each neighbor v of best_idx
        for v in neighbors[best_idx]:
            if not covered[v]:
                covered[v] = True
                # v is now covered, so it reduces gain of all u that have v in their neighborhood
                # u has v in B(u,1) iff v has u in B(v,1)
                for u in neighbors[v]:
                    gain[u] -= 1
                    # If gain[u] becomes 0, it's fully covered, but we don't need to special case it for the max scan
                    # We just need to ensure we stop when all are covered.
                    # Actually, if gain[u] is 0, it means all neighbors of u are covered.
                    # But u itself might not be covered? No, u is a neighbor of u.
                    # So if gain[u] == 0, u is covered.
        
        # Time check
        if time.time() > deadline:
            break

    # Convert code indices to word lists
    # But first, let's apply Peeling to reduce size
    
    # Peeling Phase
    # Map indices to a set for fast removal?
    # We work with the list of indices.
    
    # cover_count[i] = number of codewords c such that i in B(c, radius)
    cover_count = [0] * N
    for c_idx in code:
        for v in neighbors[c_idx]:
            cover_count[v] += 1
            
    # Iterate and remove non-essential codewords
    changed = True
    while changed:
        changed = False
        # Iterate backwards to allow removal during iteration
        # Or create a list of indices to remove
        to_remove = []
        for c_idx in code:
            # Check if c_idx is essential
            essential = False
            for v in neighbors[c_idx]:
                if cover_count[v] == 1:
                    essential = True
                    break
            if not essential:
                to_remove.append(c_idx)
                
        if to_remove:
            for c_idx in to_remove:
                code.remove(c_idx) # O(N) but N is small (~500)
                for v in neighbors[c_idx]:
                    cover_count[v] -= 1
            changed = True
            
    # Time check
    if time.time() > deadline:
        pass # Return whatever we have

    # Convert indices to lists of ints
    final_code = [idx_to_word(idx) for idx in code]
    
    return final_code
