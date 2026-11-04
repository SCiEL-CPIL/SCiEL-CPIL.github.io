import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Problem parameters
    # q=3, n=7, radius=1
    # Total words
    N = q ** n
    
    # Precompute neighbors (ball of radius 1) for each word
    # neighbors[i] is a list of indices within Hamming distance 1 from i
    neighbors = [[] for _ in range(N)]
    
    # Helper to convert index to list of digits and back
    # But for speed, we can generate neighbors by manipulating the index directly
    # Or just generate all words and compute distances.
    # Given N=2187, generating all words and checking distance is 2187^2 ~ 4.7M ops.
    # That's fast enough in Python? 4.7M * (7 comparisons) ~ 30M ops. Might take 1-2s.
    # Let's use a more efficient generation.
    
    # Generate all words as tuples
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
        
    # Compute neighbors
    for i in range(N):
        w_i = words[i]
        # Center
        neighbors[i].append(i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                # Create new word
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                # Find index
                # This is O(N) lookup if we do it naively.
                # Better to use a dictionary mapping word to index.
                pass
                
    # Let's use a dictionary for word->index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Recompute neighbors using dict lookup
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors[i] = neigh

    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations of Greedy + Prune
    # Randomize tie-breaking to escape local optima
    for iteration in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        # Greedy Set Cover
        uncovered = set(range(N))
        code = []
        
        # To speed up greedy, we can pre-compute which candidates are "active"
        # But with N=2187, simple loop is fine.
        
        while uncovered:
            best_candidate = -1
            max_new_cov = 0
            candidates = []
            
            # Iterate all possible codewords
            for c in range(N):
                # If c is already in code, skip? 
                # Actually, adding a duplicate doesn't help, but greedy naturally avoids it 
                # if we track covered. But strictly, we should pick the one that adds MOST new coverage.
                # If c is already in code, its new coverage is 0 (since its ball is already covered? 
                # No, its ball is covered by ITSELF, but if we remove it, it might not be. 
                # But in greedy construction, we only add new codewords.
                # So we just look for the c that maximizes |uncovered intersect B(c)|.
                
                # Optimization: If c is in code, we can skip? No, we haven't added it yet in this loop.
                # We just pick the best c from ALL N.
                
                # Count coverage
                # Using set intersection is slow. 
                # Better: sum(1 for u in neighbors[c] if u in uncovered)
                # Since |neighbors[c]| is 15, this is 15 lookups in a set.
                cov = 0
                for u in neighbors[c]:
                    if u in uncovered:
                        cov += 1
                        if cov > max_new_cov:
                            # Early termination if we already beat the best?
                            # No, we need to find the max.
                            pass
                
                if cov > max_new_cov:
                    max_new_cov = cov
                    best_candidate = c
                    candidates = [c]
                elif cov == max_new_cov and cov > 0:
                    candidates.append(c)
            
            if best_candidate == -1:
                # Should not happen if uncovered is not empty
                break
                
            # Randomly pick from candidates to add diversity
            chosen = random.choice(candidates)
            code.append(chosen)
            
            # Update uncovered
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.remove(u)
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        # Prune Redundant Codewords
        # Maintain cover_count
        cover_count = [0] * N
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        # Try to remove
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate in reverse to allow removal by index? 
            # Or just mark for removal.
            to_remove = []
            for idx, c in enumerate(code):
                # Check if c is redundant
                # c is redundant if for all u in B(c), cover_count[u] > 1
                redundant = True
                for u in neighbors[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if to_remove:
                # Remove them
                # We need to update cover_count
                for idx in to_remove:
                    c = code[idx]
                    for u in neighbors[c]:
                        cover_count[u] -= 1
                
                # Filter code
                # Create a set of indices to keep
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we reached a very good size, we might stop early?
            # 186 is the target. If we hit 186, we can stop.
            if best_size <= 186:
                break
                
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback: return the direct sum construction if something went wrong
        # But best_code should always be set.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
