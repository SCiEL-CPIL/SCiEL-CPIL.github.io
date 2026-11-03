import time
import random
import sys
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin from 120s limit
    
    total_words = q ** n
    
    # Precompute balls (neighborhoods) for each word
    # For radius=1, each ball has 1 + n*(q-1) words
    ball_size = 1 + n * (q - 1)
    
    # Convert word to index
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute all balls as lists of indices
    balls = []
    # Precompute coverers: for each word, which codewords cover it
    coverers = [[] for _ in range(total_words)]
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        # Update coverers
        for nb in neighbors:
            coverers[nb].append(idx)
            
    # Optimization: Convert coverers to sets or sorted lists for intersection?
    # Since we need intersection of lists, and lists are small (max 1 + n*(q-1) = 1+7*2=15 per word? No, coverers size is number of codewords covering a word.
    # In a good code, each word is covered by ~186 * 15 / 2187 ~ 1.28 times.
    # So coverers[w] is small. Intersection of small lists is fast.
    
    def greedy_cover_lazy(seed):
        """
        Lazy greedy set cover: use a priority queue where entries are (gain, codeword).
        When popping, recompute gain; if it changed, skip.
        """
        random.seed(seed)
        covered_mask = 0
        code = []
        
        # Initialize heap with all codewords and their initial gains
        heap = []
        for c in range(total_words):
            gain = len(balls[c]) # All are uncovered initially
            heap.append((-gain, c))
        heapq.heapify(heap)
        
        while covered_mask != (1 << total_words) - 1:
            if time.time() > deadline:
                return None
            
            # Pop until we find a valid entry
            valid_gain = -1
            valid_c = -1
            while heap:
                neg_gain, c = heapq.heappop(heap)
                # Recompute gain
                # Count how many neighbors of c are NOT in covered_mask
                gain = 0
                for nb in balls[c]:
                    if not (covered_mask >> nb) & 1:
                        gain += 1
                
                if gain == -neg_gain:
                    valid_gain = gain
                    valid_c = c
                    break
                # If gain changed (decreased), we discard this entry.
                # We do NOT push it back immediately, relying on the fact that
                # if it's still the best, it might have been pushed by another path?
                # No, standard lazy heap: if we pop an outdated entry, we just discard it.
                # But we must ensure the heap isn't empty if there are still valid candidates.
                # The issue: if we discard an entry, and it was the only one with high gain,
                # we might pop lower gain entries next. This is correct behavior for lazy heap.
            
            if valid_c == -1 or valid_gain <= 0:
                # Rebuild heap if empty or no valid gain
                # This shouldn't happen if we haven't covered all, but just in case
                heap = []
                for c in range(total_words):
                    gain = 0
                    for nb in balls[c]:
                        if not (covered_mask >> nb) & 1:
                            gain += 1
                    if gain > 0:
                        heap.append((-gain, c))
                heapq.heapify(heap)
                
                if not heap:
                    break
                
                # Pop again
                while heap:
                    neg_gain, c = heapq.heappop(heap)
                    gain = 0
                    for nb in balls[c]:
                        if not (covered_mask >> nb) & 1:
                            gain += 1
                    if gain == -neg_gain:
                        valid_gain = gain
                        valid_c = c
                        break
                
                if valid_c == -1 or valid_gain <= 0:
                    break
            
            # Select from top candidates with same gain for randomization
            # Collect all with same gain
            candidates = [valid_c]
            threshold = valid_gain
            while heap:
                neg_gain, c = heapq.heappop(heap)
                gain = 0
                for nb in balls[c]:
                    if not (covered_mask >> nb) & 1:
                        gain += 1
                if gain == threshold:
                    candidates.append(c)
                elif gain < threshold:
                    # Put it back
                    if gain > 0:
                        heapq.heappush(heap, (-gain, c))
                    break
                else:
                    # Shouldn't happen
                    heapq.heappush(heap, (-gain, c))
            
            chosen = random.choice(candidates)
            code.append(chosen)
            
            # Update covered_mask
            for nb in balls[chosen]:
                covered_mask |= (1 << nb)
            
            # Push back non-chosen candidates with updated gains?
            # No, lazy approach: we don't update existing heap entries.
            # We just rely on re-evaluation when they are popped.
            # However, the candidates we popped but didn't choose need to be re-inserted
            # with their CURRENT gains, otherwise they are lost.
            for c in candidates:
                if c != chosen:
                    gain = 0
                    for nb in balls[c]:
                        if not (covered_mask >> nb) & 1:
                            gain += 1
                    if gain > 0:
                        heapq.heappush(heap, (-gain, c))
                    # If gain is 0, we can discard it as it covers nothing new.

        return code
    
    def reduce_cover(code):
        """Remove redundant codewords."""
        if not code:
            return code
        
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code
    
    def local_search(code):
        """
        Local search: try to remove codewords, and if that fails, try swaps
        using the intersection of coverers to find valid replacements efficiently.
        """
        if not code:
            return code
        
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        # Precompute set of current codewords for fast lookup? 
        # in_code is a boolean list, O(1) lookup.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each codeword
            for i in range(len(code)):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: for each codeword, try to replace it with another that covers its unique words
            for i in range(len(code)):
                c = code[i]
                # Find words only covered by c
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    continue  # Should have been caught by removal
                
                # Find candidates that cover ALL unique_words
                # Candidates must be in the intersection of coverers[w] for w in unique_words
                # And must NOT be in the current code (in_code)
                
                # Start with the coverers of the first unique word
                candidates = set(coverers[unique_words[0]])
                candidates.discard(c) # c is in coverers but we don't want to swap with itself
                
                for w in unique_words[1:]:
                    # Intersect with coverers[w]
                    # Optimization: if candidates is empty, break
                    if not candidates:
                        break
                    # Efficient intersection
                    # coverers[w] is a list. candidates is a set.
                    # Iterate over smaller one? candidates is likely small.
                    # new_candidates = {x for x in candidates if x in coverers_set[w]}
                    # But coverers[w] is a list. Converting to set every time is expensive.
                    # Let's just check membership in the list? List length is small (~1-5).
                    # Or precompute coverers as sets?
                    # Let's precompute coverers_sets for faster intersection.
                    pass # Will handle in precomputation below

                # Re-do with precomputed sets if needed, or just simple loop
                # Given the small size, a simple filter is okay.
                
                valid_candidates = []
                for cand in candidates:
                    if in_code[cand]:
                        continue
                    # Check if cand covers all other unique words
                    # We already know it covers unique_words[0] from the set intersection start?
                    # Wait, the set intersection above was not fully implemented in the thought process.
                    # Let's do it properly here.
                    
                    # Actually, let's just iterate through all codewords NOT in code?
                    # That's 2187 iterations. 
                    # The intersection method:
                    # C_cand = Intersection(coverers[w] for w in U)
                    
                    # Let's re-implement the intersection logic here cleanly.
                    pass

                # Correct Intersection Logic:
                # 1. Get list of coverers for U[0]
                # 2. For each other w in U, filter the list to keep only those in coverers[w]
                
                cand_list = [x for x in coverers[unique_words[0]] if x != c and not in_code[x]]
                
                if not cand_list:
                    continue

                for w in unique_words[1:]:
                    if not cand_list:
                        break
                    # Filter cand_list to keep only those in coverers[w]
                    # coverers[w] is a list. 
                    # To make this O(1) lookup, we can use a boolean array or set.
                    # Since this is inside a tight loop, let's assume the lists are small enough 
                    # that `x in coverers[w]` is acceptable, OR we precompute sets.
                    
                    # Precomputation idea: coverer_sets = [set(lst) for lst in coverers]
                    # Then: cand_list = [x for x in cand_list if x in coverer_sets[w]]
                    
                    # Let's use the precomputed sets.
                    cand_list = [x for x in cand_list if x in coverer_sets[w]]
                
                if not cand_list:
                    continue

                # Pick a candidate
                # Heuristic: Pick one that covers the most "shared" words (words with count > 1)
                # to potentially enable further reductions.
                best_swap = -1
                best_score = -1
                
                for c_prime in cand_list:
                    # Score: number of words in ball[c_prime] that have cover_count > 1
                    # This indicates "redundancy" we might be able to exploit later?
                    # Or maybe we want to minimize the "loss" of unique coverage?
                    # Since c_prime covers all unique words of c, the coverage of U is maintained.
                    # The net change in code size is 0 (swap).
                    # We only accept the swap if it allows a subsequent removal.
                    # So we should check if the swap makes any OTHER codeword redundant.
                    
                    # Simulate swap:
                    # Remove c: cover_count[w]-- for w in ball[c]
                    # Add c_prime: cover_count[w]++ for w in ball[c_prime]
                    
                    # Check if any codeword j != c, j != c_prime becomes redundant.
                    # A codeword j becomes redundant if all its words have cover_count <= 1 (after swap).
                    # Actually, if any word has cover_count == 1, it's NOT redundant.
                    # So we need to check if there exists a j such that for all w in ball[j], new_cover_count[w] == 1.
                    
                    # This is expensive to check for all j.
                    # Heuristic: Check if c_prime covers any word that was previously covered by c 
                    # and is now "over-covered" or something?
                    
                    # Let's just pick the first valid candidate for now, or random.
                    # To improve, we can check if the swap reduces the "total cover count sum" 
                    # or something similar.
                    
                    # For now, let's just pick one and see if it helps.
                    # If we pick randomly, we might loop.
                    # Let's pick the one that maximizes the number of words with count > 1 
                    # that it covers? No, that increases redundancy.
                    # We want to create redundancy that can be resolved?
                    
                    # Actually, the standard approach is:
                    # Perform swap.
                    # Try to remove ANY codeword.
                    # If removal succeeds, keep swap.
                    # Else, revert swap.
                    
                    # We can do this trial-and-error inside the loop.
                    # But it's expensive.
                    
                    # Let's just pick the first candidate.
                    best_swap = c_prime
                    break # Take the first one found
                    
                if best_swap != -1:
                    # Perform swap
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = best_swap
                    
                    for w in balls[best_swap]:
                        cover_count[w] += 1
                    in_code[best_swap] = True
                    
                    # Check if this swap enabled a reduction
                    reduced = False
                    for j in range(len(code)):
                        if j == i: continue # Skip the swapped one? No, it might be redundant now? Unlikely.
                        cj = code[j]
                        redundant = True
                        for w in balls[cj]:
                            if cover_count[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            for w in balls[cj]:
                                cover_count[w] -= 1
                            in_code[cj] = False
                            code.pop(j)
                            reduced = True
                            break
                    
                    if reduced:
                        improved = True
                        break
                    else:
                        # Revert swap
                        for w in balls[best_swap]:
                            cover_count[w] -= 1
                        in_code[best_swap] = False
                        code[i] = c
                        for w in balls[c]:
                            cover_count[w] += 1
                        in_code[c] = True
                        
        return code

    # Precompute coverer_sets for efficient intersection in local_search
    coverer_sets = [set(lst) for lst in coverers]
    
    # Main loop
    best_solution = None
    best_size = float('inf')
    
    # First run with seed 0
    initial_code = greedy_cover_lazy(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover_lazy(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
    
    if best_solution is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
    
    return result
