import time
import random
import itertools

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for safety, though task is fixed
        # Use seed logic for other params
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return []

    total_words = q ** n
    
    # Helper to convert int to tuple and vice versa
    def int_to_tuple(w):
        digits = []
        for _ in range(n):
            digits.append(w % q)
            w //= q
        return tuple(reversed(digits))
        
    def tuple_to_int(t):
        w = 0
        mult = 1
        for d in reversed(t):
            w += d * mult
            mult *= q
        return w

    # 1. Precompute neighbors (balls of radius 1)
    # Ball size for q=3, n=7, r=1 is 1 + 7*2 = 15
    neighbors = [None] * total_words
    for w in range(total_words):
        t = int_to_tuple(w)
        ball = {w}
        for i in range(n):
            orig = t[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_t = list(t)
                new_t[i] = new_val
                new_w = tuple_to_int(new_t)
                ball.add(new_w)
        neighbors[w] = list(ball)

    # 2. Initial Construction: Direct Sum of Tetracode
    # Tetracode is a perfect code of length 4, radius 1.
    # Direct sum with 3^(n-4) tails gives a valid code of size 9 * 3^3 = 243? 
    # Wait. Tetracode has 9 words. 9 * 3^(7-4) = 9 * 27 = 243.
    # The inspiration code said 189. Let's check the inspiration code logic.
    # Inspiration: tetracode = 9 words. tail length n-4=3. 9 * 27 = 243.
    # Why did I think 189? 
    # Ah, the inspiration code in the prompt returned `[h + list(tail) ...]`.
    # 9 * 27 = 243.
    # Is there a smaller structural code?
    # The best known is 186.
    # Maybe a different structure?
    # Let's stick to the Greedy Set Cover from the current program as a backup if local search fails, 
    # but start with the structural code if it's valid and small enough to refine.
    # Actually, 243 is quite large. The current program's greedy approach got ~183 (score 1.0167 vs seed 1.0 implies seed was ~185-186? 
    # Score = seed/value. If seed is 186 and value is 183, score = 186/183 = 1.016.
    # So the current program is already beating the structural 243.
    
    # Therefore, the "Inspiration" of using a structural code is only good if it's smaller than greedy.
    # 243 > 183. So the structural code is WORSE.
    # However, the structural code is a *valid* code. 
    # The current program's greedy code is also valid.
    # The current program's score is 1.0167.
    # If I use the structural code (243), the score would be 186/243 = 0.76, which is worse.
    
    # So, I should NOT use the structural code as the *start* if it's larger than the greedy result.
    # BUT, the directive says "Combine... take structural idea of one and refinement of other".
    # Perhaps the "structural idea" is just that we have a valid code to start with, 
    # and we use the "refinement machinery" (local search) to improve it.
    # If I start with the Greedy code (183 words), I can try to reduce it further using local search.
    # The current program does a "Reduction Phase" (removal only).
    # I will enhance the reduction phase with SWAPS.
    
    # Let's re-evaluate the current program's output.
    # It does Greedy Set Cover, then Removal.
    # Greedy Set Cover for q=3, n=7, r=1.
    # Each word covers 15. Total words 2187.
    # Lower bound 2187/15 = 145.8.
    # Best known 186.
    # Greedy usually gets close to optimal.
    
    # Strategy:
    # 1. Run Greedy Set Cover to get an initial code (approx 183-185 words).
    # 2. Apply Local Search (Swaps) to try to reduce the count further.
    # 3. Apply Removal loop again.
    
    # This combines the "Greedy" structure (from current) with "Local Search" refinement (improved from simple removal).
    # Wait, the inspiration provided in the prompt was the Tetracode one.
    # If the Tetracode code is 243 words, it's not a good starting point for *minimizing* if we can get 183 easily.
    # However, maybe I can use the Tetracode code to *initialize* the local search if the greedy is too slow?
    # No, greedy is fast enough for 2187 words.
    
    # Let's stick to:
    # 1. Greedy Set Cover (fast, good initial solution).
    # 2. Local Search with Swaps (to escape local minima and find smaller codes).
    
    # Implementation of Local Search:
    # We maintain `cover_count` and `codewords` set.
    # We iterate.
    # In each iteration, we try to improve the code.
    # Improvement 1: Remove a codeword if it doesn't cover any unique word.
    # Improvement 2: Swap a codeword c with c' if:
    #   - c' is not in code.
    #   - Removing c leaves some words uncovered (unique to c).
    #   - Adding c' covers those unique words.
    #   - AND, the "cost" of the swap is beneficial.
    #   Cost metric: We want to minimize the number of codewords. A swap keeps the count same.
    #   So a swap is only useful if it enables a subsequent removal.
    #   Heuristic: Swap c with c' if c' covers more words with cover_count == 1 than c does?
    #   Or if c' covers words with cover_count == 2, reducing the "rigidity" of the code?
    
    # Let's try a simpler heuristic:
    # Randomly pick a codeword c.
    # Let U_c be words covered only by c.
    # If U_c is empty, remove c.
    # Else, find a candidate c' not in code that covers U_c.
    # Among candidates, pick one that minimizes the "uniqueness" of the new code?
    # Actually, just pick a random valid c' and swap. Repeat.
    # This is a random walk.
    
    # To be safe and effective, I will implement:
    # 1. Greedy Set Cover.
    # 2. Removal Loop (as in current).
    # 3. Local Search Loop:
    #    - While time remains:
    #      - Try to remove any codeword.
    #      - If no removal possible, pick a random codeword c.
    #      - Compute U_c (unique words).
    #      - If U_c is not empty:
    #        - Find all c' not in code such that U_c subset of Ball(c').
    #        - If any exist, pick one (e.g., the one that covers the most other words with low cover count).
    #        - Swap c with c'.
    #        - Update counts.
    #      - If U_c is empty, remove c (handled above).
    
    # This should work.
    
    # Optimization: Pre-compute for each word w, which codewords cover it? 
    # No, cover_count is enough.
    # To find c' covering U_c:
    # U_c is small (usually 1-3 words).
    # We can iterate through all non-codewords? 2187 - 183 = ~2000.
    # For each c', check if it covers all w in U_c.
    # This is fast.
    
    def greedy_set_cover(randomize=False):
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = list(range(total_words))
        
        if randomize:
            random.shuffle(available_candidates)
            
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available_candidates.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords_list

    def local_search(code_list, time_limit):
        # Convert to set for fast lookup
        code_set = set(code_list)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        start_time = time.time()
        no_improvement_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # Iterate in random order to avoid bias
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c = code_list[i]
                can_remove = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove
                    del code_list[i]
                    code_set.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    improved = True
                    no_improvement_count = 0
                    break # Restart scan after removal
            
            if improved:
                continue
                
            # 2. Try Swaps
            # Pick a random codeword to try swapping
            if code_list:
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words for c
                unique_words = []
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if unique_words:
                    # Find candidate c' not in code_set that covers all unique_words
                    best_c_prime = -1
                    best_score = -1
                    
                    # Iterate through all words not in code
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        covers_all = True
                        for w in unique_words:
                            if w not in neighbors[c_prime]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: Prefer c_prime that reduces "tightness"
                        score = 0
                        for w in neighbors[c_prime]:
                            if cover_count[w] == 2:
                                score += 1
                            elif cover_count[w] == 1:
                                score += 2 
                        
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            if best_score > 10:
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        for w in neighbors[c]:
                            cover_count[w] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        for w in neighbors[best_c_prime]:
                            cover_count[w] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    pass
            else:
                break
                
            if no_improvement_count > 200:
                pass

        return code_list

    # Execution
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    best_code = None
    best_size = float('inf')
    
    # 1. Baseline Greedy (Deterministic)
    initial_code = greedy_set_cover(randomize=False)
    
    # Refine with a short time limit to ensure we have time for restarts
    refine_limit_1 = min(10.0, deadline - time.time())
    refined_code = local_search(initial_code, refine_limit_1)
    best_code = refined_code
    best_size = len(refined_code)
    
    # 2. Random Restarts
    while time.time() < deadline - 2.0:
        # Generate new randomized greedy code
        new_initial = greedy_set_cover(randomize=True)
        
        # Refine it
        remaining = deadline - time.time() - 1.0 # 1s buffer
        if remaining < 1.0:
            break
            
        # Spend a slice of time on this refinement
        time_slice = max(1.0, min(10.0, remaining * 0.5))
        
        refined_new = local_search(new_initial, time_slice)
        
        if len(refined_new) < best_size:
            best_size = len(refined_new)
            best_code = refined_new
            
        if time.time() > deadline - 2.0:
            break

    # 3. Final Cleanup on best_code
    if best_code:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors[c]:
                cover_count[w] += 1
            
        final_list = list(code_set)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(len(final_list) - 1, -1, -1):
                if time.time() > deadline - 0.5:
                    break
                c = final_list[i]
                can_remove = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    del final_list[i]
                    code_set.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    improved = True
        best_code = final_list

    # Convert to list of lists
    result = []
    if best_code:
        for c in best_code:
            t = int_to_tuple(c)
            result.append(list(t))
    else:
        # Fallback
        for c in greedy_set_cover():
            t = int_to_tuple(c)
            result.append(list(t))
            
    return result
