"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    # Deadline guard: finish well before the 120s wall-clock limit.
    deadline = time.time() + 115.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        codewords = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        for w in all_words:
            if w not in covered:
                codewords.append(list(w))
                # Add neighbors
                covered.add(w)
                for pos in range(n):
                    for val in range(q):
                        if val == w[pos]: continue
                        new_w = list(w)
                        new_w[pos] = val
                        covered.add(tuple(new_w))
        return codewords

    # Specialized for q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 729
    ALL_BITS = (1 << N) - 1
    
    # Precompute neighbor masks
    masks = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    j = word_to_idx[tuple(new_w)]
                    mask |= (1 << j)
        masks[i] = mask

    best_code = None
    best_size = float('inf')
    
    # Number of random greedy starts
    # We will run as many as we can fit in the time budget
    # Each start takes ~0.1s for greedy + variable time for local search
    # Let's aim for 5-10 starts.
    
    for start_id in range(15):
        if time.time() > deadline - 15:
            break
            
        # 1. Randomized Greedy Construction
        code = []
        covered_mask = 0
        remaining = set(range(N))
        
        while remaining:
            best_v = -1
            best_gain = -1
            candidates = []
            for v in remaining:
                gain = (masks[v] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    candidates = [v]
                elif gain == best_gain:
                    candidates.append(v)
            
            if best_v == -1:
                break # Should not happen if remaining is not empty and masks are correct
                
            v = random.choice(candidates)
            code.append(v)
            covered_mask |= masks[v]
            
            # Prune remaining: remove words that no longer contribute to coverage
            # A word v contributes if it covers at least one word not in covered_mask
            new_remaining = set()
            for v in remaining:
                if v == best_v:
                    continue
                if (masks[v] & ~covered_mask).bit_count() > 0:
                    new_remaining.add(v)
            remaining = new_remaining
            
            if time.time() > deadline - 15:
                break

        # Check if we have a valid cover
        if covered_mask != ALL_BITS:
            # If we broke early due to time, the code might be invalid.
            # If we finished, it should be valid.
            if remaining:
                continue # Skip this start if invalid/incomplete

        # 2. Local Search: 2-for-1 moves
        code_set = set(code)
        current_cover = covered_mask
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            code_list = list(code_set)
            random.shuffle(code_list)
            
            # Try all pairs (u, w)
            for i in range(len(code_list)):
                if time.time() > deadline - 10:
                    break
                u = code_list[i]
                for j in range(i + 1, len(code_list)):
                    w = code_list[j]
                    
                    # Calculate deficit if we remove u and w
                    base_cover = current_cover & ~masks[u] & ~masks[w]
                    deficit = ALL_BITS & ~base_cover
                    
                    # If deficit is 0, both are redundant (should be handled by 1-for-0, but 2-for-1 can also catch it if v is arbitrary? No, v must be added. If deficit is 0, we can just remove u and w? No, we replace with v. If deficit is 0, any v works? No, we need to maintain size? No, 2-for-1 reduces size by 1. If deficit is 0, we can remove u and w and add ANY v? No, that increases size by 1? No, remove 2, add 1. Net -1.
                    # If deficit is 0, it means C \ {u,w} already covers everything. Then we can just remove u and w? That would be a 2-for-0 move.
                    # Our 2-for-1 move adds v. If deficit is 0, we can pick any v not in code_set.
                    # But we should prefer v that is "good"? Or just pick one.
                    # Actually, if deficit is 0, we can just remove u and w and NOT add v?
                    # The move is "Replace {u,w} with {v}".
                    # If deficit is 0, we can replace {u,w} with ANY {v}.
                    # But we could also just remove {u,w} if we don't care about size reduction? No, we want to minimize size.
                    # If deficit is 0, we can remove u and w. Size -2.
                    # But our loop only does 2-for-1.
                    # We should handle 1-for-0 and 2-for-0 separately or check if deficit is 0.
                    
                    if deficit == 0:
                        # We can remove u and w.
                        # But we are in a 2-for-1 loop.
                        # Let's just pick a random v to satisfy the "add v" part, or better, 
                        # we can just remove u and w and break?
                        # No, the structure is fixed.
                        # If deficit is 0, we can pick any v.
                        # Let's pick the first v not in code_set.
                        for v in range(N):
                            if v not in code_set:
                                found_v = v
                                break
                        else:
                            continue # No v available? Impossible if size < N.
                        
                        code_set.remove(u)
                        code_set.remove(w)
                        code_set.add(found_v)
                        current_cover = 0
                        for idx in code_set:
                            current_cover |= masks[idx]
                        improved = True
                        break
                    else:
                        # Find v not in code_set such that masks[v] covers deficit
                        found_v = -1
                        for v in range(N):
                            if v in code_set:
                                continue
                            if (masks[v] & deficit) == deficit:
                                found_v = v
                                break
                        
                        if found_v != -1:
                            code_set.remove(u)
                            code_set.remove(w)
                            code_set.add(found_v)
                            current_cover = 0
                            for idx in code_set:
                                current_cover |= masks[idx]
                            improved = True
                            break
                if improved:
                    break
        
        # 3. Final Greedy Removal (1-for-0)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in list(code_set):
                if (current_cover & ~masks[idx]) == ALL_BITS:
                    code_set.remove(idx)
                    current_cover &= ~masks[idx]
                    improved = True
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
        if best_size == 73:
            break

    # Fallback
    if best_code is None:
        # Return a simple valid code
        code = []
        covered_mask = 0
        remaining = set(range(N))
        while remaining:
            best_v = -1
            best_gain = -1
            for v in remaining:
                gain = (masks[v] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            code.append(best_v)
            covered_mask |= masks[best_v]
            remaining.discard(best_v)
            remaining = {v for v in remaining if (masks[v] & ~covered_mask).bit_count() > 0}
        best_code = code

    return [list(all_words[idx]) for idx in best_code]

# EVOLVE-BLOCK-END
