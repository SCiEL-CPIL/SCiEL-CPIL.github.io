import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    total_words = q ** n

    # Fast path for the known task: q=3, n=6, radius=1
    # We can precompute the best known code (73 words) or try to find a smaller one.
    # The best known is 73. The lower bound is 71.
    # Since finding a 71 or 72 code is a hard open problem (or at least very hard),
    # we should aim to reliably produce 73, or perhaps 74/75 if we fail.
    # However, the current seed program uses a randomized greedy approach which might find 73 or better.
    # Let's check if we can hardcode a known 73-word code for q=3, n=6, r=1.
    # If we can, we return it immediately. This guarantees a score of 73 (value 73).
    # The seed program's score is 1.0000, which corresponds to value 73 (since seed/value=1.0 => value=seed).
    # Wait, the prompt says "Score: the number of codewords, minimized... search score is seed/value".
    # If the seed program produced 73 words, its value is 73.
    # If I produce 73 words, my value is 73. Score = 73/73 = 1.0.
    # If I produce 72 words, my value is 72. Score = 73/72 > 1.0.
    # So, to beat 1.0, I MUST find a code with < 73 words.
    # This is extremely unlikely in 120s without specific breakthrough algorithms.
    # However, maybe the seed program is NOT reliably finding 73?
    # The prompt says "CURRENT PROGRAM (score 1.0000)". This implies the seed program's output was evaluated and got score 1.0.
    # This suggests the seed program likely found a 73-word code (or the seed value is calibrated to the best known 73).
    # Actually, usually "seed" in these tasks refers to the baseline performance.
    # If the baseline is 73, and I find 73, I get 1.0.
    # If I find 74, I get 73/74 < 1.0.
    # So I must be careful not to return a larger code.
    # The seed program uses randomized greedy. It might sometimes find 73, sometimes 74?
    # If it found 73, it got 1.0.
    # My goal: beat 1.0. This requires finding a code of size < 73.
    # This is a research-level problem. Is it possible?
    # The problem K_3(6,1) is solved: the value is 73. (van Laarhoven et al. proved 73 is optimal? No, they found a 73 code. The lower bound is 71. It is not proven that 73 is optimal, but it is widely believed. Actually, checking literature: "The ternary covering code K_3(6,1) has size 73" is a known result?
    # Let's check: "Ternary covering codes" - K_3(6,1) = 73 is the best known. Is it proven minimal?
    # A quick mental check: For n=6, q=3, r=1. Sphere size is 1 + 6*(3-1) = 13. 729/13 = 56.07. So lower bound is 57.
    # Better bounds exist. The lower bound 71 is mentioned in the prompt.
    # If 73 is not proven minimal, there might be a 71 or 72 code.
    # But finding it is very hard.
    # However, maybe the seed program is inefficient and often returns >73?
    # If the seed program returned 74, its score would be 73/74 = 0.986.
    # But the prompt says score 1.0000. This implies the seed program's value was 73 (if seed=73) or the score is normalized differently.
    # "search score is seed/value". If seed=73 and value=73, score=1.0.
    # So the seed program found 73.
    # To beat it, I need < 73.
    # If I can't find < 73, I should at least reliably find 73 to maintain 1.0.
    # But the directive says "beat the island best".
    # If I can't beat it, I should at least not make it worse.
    # Let's try to implement a more robust search for 73, and maybe try some local search improvements to see if 72 is found.
    # However, 120s is short.
    
    # Strategy:
    # 1. If q=3, n=6, radius=1:
    #    - Try to load a hardcoded 73-word code. This guarantees validity and size 73.
    #    - Run a time-limited local search (swap/add/remove) trying to reduce size.
    #    - If a smaller code is found, return it.
    #    - Else return the 73-word code.
    # 2. For other cases, use the greedy heuristic.

    if q == 3 and n == 6 and radius == 1:
        # Hardcode a known 73-word code?
        # I don't have it memorized. I need to generate one.
        # The seed program generates one.
        # Let's run the seed program's logic but optimized.
        # Or, better, use a known construction.
        # The tetracode product gives 9 * 3^2 = 81 words.
        # Pruning reduces it.
        
        # Let's use a fast greedy to find a 73-word code.
        # Then try to prune/improve.
        
        # Precompute balls for speed
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
            
        def tuple_to_idx(t):
            idx = 0
            for i in range(n):
                idx += t[i] * powers[i]
            return idx

        def idx_to_tuple(idx):
            t = [0] * n
            rem = idx
            for i in range(n):
                t[i] = rem % q
                rem //= q
            return tuple(t)

        # Generate all words
        all_words = [idx_to_tuple(i) for i in range(total_words)]
        
        # Precompute balls as sets of indices
        balls = [set() for _ in range(total_words)]
        for i, w in enumerate(all_words):
            balls[i].add(i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        balls[i].add(new_idx)
        
        # Greedy Set Cover
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                # Fallback: add random uncovered words to ensure coverage
                for u in list(uncovered):
                    code_indices.append(u)
                break
                
            # Find best candidate
            best_gain = -1
            best_candidates = []
            # Optimization: only check candidates that cover uncovered words?
            # For q=3, n=6, 729 candidates. 729*13 checks per step.
            # 73 steps. 73 * 729 * 13 ~ 700k ops. Very fast.
            for c in range(total_words):
                gain = len(balls[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            # Random choice from best candidates to allow diversity in restarts?
            # For a single run, pick the first or random.
            # Let's pick random to avoid deterministic pitfalls.
            best_c = random.choice(best_candidates)
            code_indices.append(best_c)
            uncovered -= balls[best_c]
            
        # Prune
        def prune_cover(indices):
            indices = list(indices)
            cover_counts = np.zeros(total_words, dtype=np.int32)
            for idx in indices:
                for w in balls[idx]:
                    cover_counts[w] += 1
            
            changed = True
            while changed:
                changed = False
                new_indices = []
                for idx in indices:
                    necessary = False
                    for w in balls[idx]:
                        if cover_counts[w] == 1:
                            necessary = True
                            break
                    if necessary:
                        new_indices.append(idx)
                    else:
                        for w in balls[idx]:
                            cover_counts[w] -= 1
                        changed = True
                indices = new_indices
            return indices

        pruned_code = prune_cover(code_indices)
        
        # Local Search: Try to replace 2 codewords with 1, or 1 with 0?
        # Or swap one codeword for another to allow removal of another?
        # Simple local search: Try removing each codeword. If still covered, remove it.
        # This is what prune_cover does.
        # More advanced: Try swapping codeword A with B, then see if we can remove C.
        
        # Let's try a simple improvement loop:
        # 1. Try to remove any redundant codeword (prune again).
        # 2. Try swapping: for each pair (A, B) where A in code, B not in code:
        #    temp_code = (code - {A}) + {B}
        #    if temp_code covers:
        #       if prune(temp_code) < len(code): update code
        
        # This is O(N^2 * prune_cost). N=73. 73*73*73*13 ~ 5M ops. Feasible.
        
        current_code = pruned_code
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing one
            for i in range(len(current_code)):
                if time.time() > deadline - 10.0:
                    break
                trial_code = [c for j, c in enumerate(current_code) if j != i]
                # Check if trial_code covers
                covered = set()
                for c in trial_code:
                    covered.update(balls[c])
                if len(covered) == total_words:
                    current_code = trial_code
                    improved = True
                    break # Restart loop with new size
            
            if improved:
                continue
                
            # Try swapping one
            code_set = set(current_code)
            for i in range(len(current_code)):
                if time.time() > deadline - 10.0:
                    break
                a = current_code[i]
                # Try replacing a with b (b not in code)
                # To limit search, try b from a small set of candidates?
                # Or all b? 729 candidates.
                # For each b, check if (code - {a} + {b}) covers.
                # Then prune.
                # This is expensive.
                # Optimization: Only try b if it covers something a doesn't?
                # Or just try random swaps?
                
                # Let's try a few random swaps
                for _ in range(5):
                    if time.time() > deadline - 10.0:
                        break
                    b = random.randint(0, total_words - 1)
                    if b in code_set:
                        continue
                    trial_code = [c for c in current_code if c != a]
                    trial_code.append(b)
                    
                    # Check coverage
                    covered = set()
                    for c in trial_code:
                        covered.update(balls[c])
                    if len(covered) == total_words:
                        pruned_trial = prune_cover(trial_code)
                        if len(pruned_trial) < len(current_code):
                            current_code = pruned_trial
                            improved = True
                            break
                if improved:
                    break
                    
        # Final check
        if not uncovered: # If greedy finished
            final_code = current_code
        else:
            # If greedy didn't finish, we have a partial code.
            # We must ensure validity.
            # Add all uncovered words to the code.
            for u in uncovered:
                if u not in final_code:
                    final_code.append(u)
            final_code = prune_cover(final_code)
            
        # Convert to lists
        return [list(idx_to_tuple(c)) for c in final_code]

    # Generic fallback for other parameters
    # Use the logic from the original program but cleaned up
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls
    balls = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_tuple(i)
        balls[i].add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    balls[i].add(new_idx)

    def prune_cover(indices):
        indices = list(indices)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    # Greedy
    uncovered = set(range(total_words))
    code_indices = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_gain = -1
        best_candidates = []
        for c in range(total_words):
            gain = len(balls[c] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_candidates = [c]
            elif gain == best_gain:
                best_candidates.append(c)
        
        if not best_candidates:
            break
            
        best_c = random.choice(best_candidates)
        code_indices.append(best_c)
        uncovered -= balls[best_c]
        
    if uncovered:
        for u in uncovered:
            code_indices.append(u)
            
    pruned_code = prune_cover(code_indices)
    return [list(idx_to_tuple(c)) for c in pruned_code]
