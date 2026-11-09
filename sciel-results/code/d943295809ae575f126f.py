import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code for q=3, n=6, radius=1.
    Uses Greedy Set Cover followed by Simulated Annealing / Local Search
    to minimize the number of codewords.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters if needed, though task is specific
        # For now, just return a trivial or random valid code if possible
        # But the task is specific to 3, 6, 1.
        pass

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = q ** n
    # Precompute spheres
    # Map word value (int) to list of words in its sphere
    spheres = [[] for _ in range(total_words)]
    
    # Helper to convert int to tuple and back
    def to_tuple(val):
        return tuple((val // (q**i)) % q for i in range(n))
        
    def to_val(tup):
        return sum(tup[i] * (q**i) for i in range(n))

    # Precompute neighbors
    for w in range(total_words):
        tw = to_tuple(w)
        spheres[w].append(w)
        for i in range(n):
            for d in range(q):
                if d == tw[i]: continue
                new_t = list(tw)
                new_t[i] = d
                spheres[w].append(to_val(tuple(new_t)))
        # Remove duplicates if any (shouldn't be any for q>2)
        spheres[w] = list(set(spheres[w]))

    # 1. Greedy Construction
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=int)
    
    while uncovered.any():
        best_c = -1
        best_count = -1
        # To speed up, we can sample or iterate all. 729 is small.
        for c in range(total_words):
            if c in code: # Optimization: skip if already in code? 
                # Actually, we can just check coverage gain.
                pass
            # Count how many uncovered words are in spheres[c]
            # sum(uncovered[i] for i in spheres[c])
            gain = np.sum(uncovered[spheres[c]])
            if gain > best_count:
                best_count = gain
                best_c = c
        
        if best_c == -1: break
        
        code.append(best_c)
        cover_count[spheres[best_c]] += 1
        uncovered[spheres[best_c]] = False
        
        if time.time() > deadline:
            break

    # Ensure we have a valid code
    # cover_count > 0 everywhere?
    if not (cover_count > 0).all():
        # Fallback: add more until valid
        while not (cover_count > 0).all():
            # Find uncovered
            unc_idx = np.where(cover_count == 0)[0]
            # Pick a random uncovered word, find a codeword covering it
            w = unc_idx[0]
            # Find any c such that w in spheres[c]
            # This is slow. Better to just add randoms or use precomputed inverse.
            # For now, just add random until valid.
            c = random.randint(0, total_words - 1)
            if c not in code:
                code.append(c)
                cover_count[spheres[c]] += 1
            else:
                # If c already in code, it must cover w? No.
                # We need a c that covers w.
                # Precompute inverse spheres?
                pass
            # Simple fix: if we are here, greedy failed or time ran out.
            # Let's just force validity by adding random codewords.
            # This is a safety net.
            if time.time() > deadline:
                break

    # 2. Local Search / Simulated Annealing
    # We want to minimize len(code)
    # Moves:
    # A. Remove c if redundant (all words in S(c) have cover_count > 1)
    # B. Swap c with c' if valid
    
    # Helper: Check if removing c is valid
    def can_remove(c):
        # Check if any word in S(c) has cover_count == 1
        # If yes, it's the only one covering it, so we can't remove c.
        # We need cover_count[w] > 1 for all w in S(c).
        # Actually, if cover_count[w] == 1, it means ONLY c covers w (since we are checking removal of c, 
        # the count includes c. So if count is 1, removing c makes it 0 -> invalid).
        # So we need min(cover_count[S(c)]) > 1.
        return np.min(cover_count[spheres[c]]) > 1

    # Helper: Check if swap c -> c' is valid
    def can_swap(c, c_new):
        # Temporarily remove c
        # Check if any word in S(c) has cover_count == 1 (before adding c_new)
        # But c_new might cover some of them.
        # So:
        # 1. Decrement cover_count for S(c)
        # 2. Increment cover_count for S(c_new)
        # 3. Check if any cover_count == 0
        # 4. Revert
        
        # Optimization: 
        # Words that become 0 are those in S(c) that had count 1 AND are not in S(c_new).
        # Let's just do the array ops, it's fast enough for 13 elements.
        
        old_counts_c = cover_count[spheres[c]].copy()
        old_counts_new = cover_count[spheres[c_new]].copy()
        
        cover_count[spheres[c]] -= 1
        cover_count[spheres[c_new]] += 1
        
        valid = np.min(cover_count) > 0
        
        cover_count[spheres[c]] = old_counts_c
        cover_count[spheres[c_new]] = old_counts_new
        
        return valid

    # Main Loop
    improved = True
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = True
        while removed_any:
            removed_any = False
            # Shuffle to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                if can_remove(c):
                    # Remove
                    cover_count[spheres[c]] -= 1
                    code.pop(idx)
                    removed_any = True
                    break # Restart scan
            
        # Phase 2: Swaps
        # Try to find a swap that allows further removals or just maintains size but improves structure
        # We will try random swaps.
        # If we find a valid swap, we accept it.
        # To escape local minima, we might accept swaps that don't immediately allow removal,
        # but since size doesn't change, we need a reason to accept.
        # Usually, we just accept any valid swap to explore the space of valid codes of the same size.
        
        if len(code) == 0: break
        
        # Pick a random codeword to swap
        idx = random.randint(0, len(code) - 1)
        c = code[idx]
        
        # Generate candidate c_new
        # Heuristic: Pick a word w in S(c) that is "weakly" covered (count is low)
        # and pick a c_new that covers it.
        # Or just random c_new.
        # Let's try a few random c_new.
        
        found_swap = False
        for _ in range(10): # Try 10 random swaps
            c_new = random.randint(0, total_words - 1)
            if c_new == c:
                continue
            if c_new in code:
                continue
            
            if can_swap(c, c_new):
                # Accept swap
                cover_count[spheres[c]] -= 1
                cover_count[spheres[c_new]] += 1
                code[idx] = c_new
                found_swap = True
                break
        
        # If no swap found, we might be stuck.
        # We can try a "kick" by adding a random codeword, removing a different one, etc.
        # But for now, let's just loop.
        
        # Occasionally, try to add a codeword and then remove 2?
        # This is complex. The swap + removal loop is standard.

    return [to_tuple(c) for c in code]
