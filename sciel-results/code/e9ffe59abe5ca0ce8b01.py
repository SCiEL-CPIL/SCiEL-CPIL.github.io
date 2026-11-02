import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Precompute covers for all possible codewords
    # Total words = q**n
    total_words = q ** n
    # Map word index to tuple
    # We need to generate all words and their covers
    
    # Efficiently generate covers
    # A word is covered by c if dist(word, c) <= radius
    # For radius 1, c covers c and its 2*n neighbors
    
    covers = [0] * total_words
    
    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        rem = i
        for _ in range(n):
            w.append(rem % q)
            rem //= q
        words.append(w)
        
    # Build covers
    # For each center c, find all words within distance radius
    # This is O(total_words * total_words) if naive. 2187^2 ~ 4.7M.
    # Distance calc is O(n). 4.7M * 7 ~ 33M ops. Fast enough in Python? Maybe 1-2s.
    # Better: Generate neighbors directly.
    
    for c_idx in range(total_words):
        c = words[c_idx]
        mask = 1 << c_idx
        # Neighbors
        for pos in range(n):
            orig_val = c[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_w = c.copy()
                new_w[pos] = new_val
                # Find index of new_w
                # Convert back to index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_w[j]
                # Wait, my index generation was little-endian (rem % q).
                # So index = sum(w[j] * q**j).
                # Let's re-verify index generation.
                # i=0 -> [0,0,0]
                # i=1 -> [1,0,0]
                # So idx = w[0] + w[1]*q + ...
                
                # Recalculate idx properly
                idx = 0
                mult = 1
                for j in range(n):
                    idx += new_w[j] * mult
                    mult *= q
                mask |= 1 << idx
        covers[c_idx] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Greedy Construction
    uncovered = full_mask
    code = []
    
    while uncovered != 0:
        if time.time() > deadline:
            break
        best_c = -1
        best_count = -1
        # Find c maximizing popcount(covers[c] & uncovered)
        # Iterate all c
        for c in range(total_words):
            count = (covers[c] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_c = c
        if best_count == 0:
            break # Should not happen if uncovered != 0
        code.append(best_c)
        uncovered &= ~covers[best_c]
        
    if not code:
        return [words[0]]
        
    # Optimization Loop
    # Current coverage
    coverage = 0
    for c in code:
        coverage |= covers[c]
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Deletion Pass
        new_code = []
        for c in code:
            # Check if c is redundant
            # coverage_without_c = union of covers of all other c's
            # Instead of recomputing full union, we can check if 
            # (coverage ^ covers[c]) == coverage ? No.
            # We need to know if covers[c] is subset of union of others.
            # Let's compute union of others.
            others_cov = 0
            for other in code:
                if other != c:
                    others_cov |= covers[other]
            if (covers[c] & ~others_cov) == 0:
                # c is redundant
                improved = True
                continue
            new_code.append(c)
        if len(new_code) < len(code):
            code = new_code
            # Recalculate coverage
            coverage = 0
            for c in code:
                coverage |= covers[c]
            continue # Restart loop as structure changed significantly
        
        # 2. Pairwise Replacement Pass
        # To speed up, we can limit checks or use a heuristic.
        # But let's try full pairwise for small codes.
        # Code size ~100. Pairs ~5000.
        # For each pair, check if replacement possible.
        
        # Precompute unique coverage for each pair relative to rest?
        # That's expensive.
        # Let's iterate pairs.
        
        # Optimization: Only consider pairs where the union of their unique covers is small.
        # We can approximate by checking if they are "close" in some metric?
        # Or just do it.
        
        # To avoid O(N^2 * N_total) which is 100^2 * 2187 = 20M ops per pass.
        # 20M bit ops in Python is slow (~20s).
        # We need to be careful.
        
        # Alternative: Randomized local search.
        # Pick random pairs.
        
        # Let's do a limited number of random pair checks per iteration.
        # Or, if code is small (< 50), do full.
        
        if len(code) < 50:
            pairs_to_check = [(i, j) for i in range(len(code)) for j in range(i+1, len(code))]
        else:
            # Sample 1000 random pairs
            pairs_to_check = []
            for _ in range(1000):
                i = np.random.randint(len(code))
                j = np.random.randint(len(code))
                if i != j:
                    pairs_to_check.append((min(i,j), max(i,j)))
            # Remove duplicates
            pairs_to_check = list(set(pairs_to_check))
            
        for i, j in pairs_to_check:
            if time.time() > deadline:
                break
            c1 = code[i]
            c2 = code[j]
            
            # Calculate coverage of rest
            rest_cov = 0
            for k in range(len(code)):
                if k != i and k != j:
                    rest_cov |= covers[code[k]]
                    
            unique_to_pair = (covers[c1] | covers[c2]) & ~rest_cov
            
            if unique_to_pair == 0:
                # Redundant, should have been caught by deletion, but maybe deletion didn't run or missed?
                # If unique_to_pair is 0, we can remove one of them.
                # Let's just remove c2 (or c1)
                code.pop(j)
                improved = True
                break # Restart loop
            
            # Check if unique_to_pair can be covered by a single codeword
            # If popcount > 15 (max ball size), impossible
            if unique_to_pair.bit_count() > 2 * n: # radius 1, ball size 1 + 2*n
                continue
                
            # Search for c_new
            found = False
            # Iterate all possible centers
            # Optimization: Only check centers that cover at least one word in unique_to_pair?
            # Actually, we need c_new such that unique_to_pair & ~covers[c_new] == 0
            
            # To speed up, we can precompute a list of centers for each word?
            # No, let's just iterate.
            # But 2187 iterations per pair is slow.
            
            # Heuristic: Check neighbors of c1 and c2 first?
            # Or just check all, but break early if found.
            
            # Let's try to find any c_new.
            # We can iterate through all c_new.
            for c_new in range(total_words):
                if (unique_to_pair & ~covers[c_new]) == 0:
                    # Replace c1, c2 with c_new
                    # Remove c2, replace c1 with c_new
                    code[i] = c_new
                    code.pop(j)
                    improved = True
                    found = True
                    break
            if found:
                break # Restart loop with new code
        
    # Final cleanup: ensure distinct and valid
    # The code should be distinct by construction (greedy adds distinct, replacement adds one distinct to pair of distincts? 
    # If c_new == c1, then we replaced {c1, c2} with {c1}. c1 was in code, c2 removed. Distinctness maintained if c_new not already in code?
    # Wait. If c_new is already in code (and not c1 or c2), then we have duplicate.
    # We must ensure c_new is not in code (excluding c1, c2).
    
    # Let's add a check in the replacement logic.
    # Actually, if c_new is already in code, say c3, then Ball(c1) U Ball(c2) subset Ball(c3) U Rest.
    # Then we can just remove c1 and c2, and keep c3.
    # So if c_new is in code, we just remove c1 and c2.
    
    # Revised Replacement Logic:
    # If found c_new:
    #   if c_new in code (and c_new != c1 and c_new != c2):
    #       remove c1, c2
    #   else:
    #       replace c1 with c_new, remove c2
    
    # Let's re-run optimization with this fix.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Deletion Pass
        new_code = []
        for c in code:
            others_cov = 0
            for other in code:
                if other != c:
                    others_cov |= covers[other]
            if (covers[c] & ~others_cov) == 0:
                improved = True
                continue
            new_code.append(c)
        if len(new_code) < len(code):
            code = new_code
            continue
        
        # 2. Pairwise Replacement Pass
        if len(code) < 50:
            pairs_to_check = [(i, j) for i in range(len(code)) for j in range(i+1, len(code))]
        else:
            pairs_to_check = []
            for _ in range(1000):
                i = np.random.randint(len(code))
                j = np.random.randint(len(code))
                if i != j:
                    pairs_to_check.append((min(i,j), max(i,j)))
            pairs_to_check = list(set(pairs_to_check))
            
        for i, j in pairs_to_check:
            if time.time() > deadline:
                break
            c1 = code[i]
            c2 = code[j]
            
            rest_cov = 0
            for k in range(len(code)):
                if k != i and k != j:
                    rest_cov |= covers[code[k]]
                    
            unique_to_pair = (covers[c1] | covers[c2]) & ~rest_cov
            
            if unique_to_pair == 0:
                code.pop(j)
                improved = True
                break
            
            if unique_to_pair.bit_count() > 2 * n:
                continue
                
            found = False
            for c_new in range(total_words):
                if (unique_to_pair & ~covers[c_new]) == 0:
                    # Check if c_new is already in code
                    if c_new in code:
                        # Just remove c1 and c2
                        # Ensure we remove indices correctly
                        if i > j:
                            i, j = j, i
                        code.pop(j)
                        code.pop(i)
                    else:
                        # Replace c1 with c_new, remove c2
                        code[i] = c_new
                        code.pop(j)
                    improved = True
                    found = True
                    break
            if found:
                break

    # Convert indices back to words
    result = [words[c] for c in code]
    # Ensure distinct
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
            
    return final_result
