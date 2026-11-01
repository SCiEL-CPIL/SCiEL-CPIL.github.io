"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Specific optimization for the known task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
        
    # Fallback for other parameters: simple greedy + pruning
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball_masks: for each word index, bitmask of indices within Hamming distance 1
    ball_masks = [0] * N
    
    # Precompute digits for each index to speed up perturbation/repair if needed, 
    # but we mostly use indices.
    
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask
    
    # Helper: Greedy Set Cover initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1  # Full mask
        selected = []
        selected_set = set()
        
        while uncovered:
            best_idx = -1
            best_count = -1
            
            # Randomized scan to break ties and explore different paths
            # We don't need to shuffle the whole list every time if we just pick randomly among ties?
            # Shuffling is O(N log N). N=2187. 
            # Alternative: Just scan 0..N-1 and keep track of best. 
            # To randomize, we can add small random noise to the count? 
            # Or just shuffle a copy of indices.
            
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for i in candidates:
                if i in selected_set:
                    continue
                    
                # Count uncovered bits covered by i
                cnt = (uncovered & ball_masks[i]).bit_count()
                
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break # Should not happen if uncovered.any()
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected

    # Helper: Local Search with Perturbation (Iterated Local Search)
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        
        # cnt[i] is the number of codewords in 'code' that cover word i
        cnt = [0] * N
        for c in code:
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb

        best_code = code.copy()
        best_size = len(code)
        no_improve_count = 0
        max_no_improve = 20 # Trigger perturbation after this many failed deletion/swap cycles
        
        while time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Deletion
            # Try to remove any codeword that doesn't uniquely cover any word
            to_remove = []
            for c in code:
                mask = ball_masks[c]
                m = mask
                is_essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                if not is_essential:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb
                    
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                improved = True
                no_improve_count = 0
                continue # Restart loop as counts changed significantly
            
            # Phase 2: Swap
            # Try to swap a codeword c with c'
            # c' must cover all words that c uniquely covers.
            
            if not code:
                break
            
            # Try a limited number of swaps per iteration to avoid infinite loops
            swaps_done = 0
            max_swaps = 50
            
            while swaps_done < max_swaps and time.time() < deadline - 2.0:
                swaps_done += 1
                c = rng.choice(code)
                
                # Find words uniquely covered by c
                unique_words = []
                mask = ball_masks[c]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        unique_words.append(idx)
                    m ^= lsb
                
                if not unique_words:
                    # Redundant, should have been caught in deletion, but handle just in case
                    code.remove(c)
                    code_set.remove(c)
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    improved = True
                    break # Restart main loop
                
                # Find candidates c' that cover all unique_words
                # Intersection of balls of unique_words
                cand_mask = ball_masks[unique_words[0]]
                for w in unique_words[1:]:
                    cand_mask &= ball_masks[w]
                    if not cand_mask:
                        break
                
                if not cand_mask:
                    continue
                
                # Filter out candidates already in code
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx not in code_set:
                        candidates.append(idx)
                    m ^= lsb
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c_prime = rng.choice(candidates)
                
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                
                code.append(c_prime)
                code_set.add(c_prime)
                m = ball_masks[c_prime]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] += 1
                    m ^= lsb
                
                improved = True
                # We don't break here to allow multiple swaps in one "pass"
                # But we restart the main loop if we want to re-evaluate deletions?
                # Actually, a swap keeps size constant. It helps enable future deletions.
                # Let's do a few swaps then re-check deletions.
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Phase 3: Perturbation (Iterated Local Search)
            if no_improve_count >= max_no_improve:
                # Perturb the current solution to escape local minimum
                # Remove 2-3 random codewords
                num_remove = rng.randint(2, 3)
                if len(code) > num_remove:
                    to_perturb = rng.sample(code, num_remove)
                    for c in to_perturb:
                        code.remove(c)
                        code_set.remove(c)
                        m = ball_masks[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb
                    
                    # Repair: Greedily add codewords to cover any uncovered words
                    # Check if any word is uncovered
                    # We can check by seeing if any cnt[i] == 0
                    # But iterating N=2187 is fast.
                    # Or we can just run a few steps of greedy on the uncovered set.
                    
                    # Find uncovered words
                    uncovered_mask = 0
                    for i in range(N):
                        if cnt[i] == 0:
                            uncovered_mask |= (1 << i)
                            
                    # Greedy repair
                    while uncovered_mask and time.time() < deadline - 2.0:
                        best_idx = -1
                        best_count = -1
                        # Randomized scan
                        candidates = list(range(N))
                        rng.shuffle(candidates)
                        for i in candidates:
                            if i in code_set:
                                continue
                            cnt_cover = (uncovered_mask & ball_masks[i]).bit_count()
                            if cnt_cover > best_count:
                                best_count = cnt_cover
                                best_idx = i
                                
                        if best_idx == -1 or best_count == 0:
                            break
                            
                        code.append(best_idx)
                        code_set.add(best_idx)
                        m = ball_masks[best_idx]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] += 1
                            m ^= lsb
                            
                        uncovered_mask &= ~ball_masks[best_idx]
                    
                    # Reset no_improve count after perturbation
                    no_improve_count = 0
                    # Note: Size might have increased. We keep best_code which is smaller.
                    # We continue searching from this perturbed state.

        return best_code

    # Main Loop: Multi-start
    best_solution = None
    best_size = float('inf')
    
    seed = 42
    while time.time() < deadline - 2.0:
        # Get initial greedy solution
        init_code = greedy_init(seed)
        
        if not init_code:
            break
            
        # Refine with local search
        refined_code = local_search(init_code, seed)
        
        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_solution = refined_code.copy()
            
        seed += 1
        
        # Early exit if we hit the theoretical lower bound? 
        # Lower bound is 156. Best known 186.
        # If we hit 156, we can stop.
        if best_size <= 156:
            break

    if best_solution is None:
        # Fallback: return empty or a trivial valid code?
        # Must return a valid code.
        # If greedy failed, return all words (valid but huge).
        # But greedy should not fail.
        return []
    
    # Convert indices to digit lists
    result = []
    for idx in best_solution:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)
        
    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback for other parameters
    # Generate all words
    all_words = []
    for i in range(q**n):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_words.append(tuple(digits))
    
    # Precompute distances or balls
    # For general case, we might not fit in bitsets if q^n is large.
    # But for this task, we assume q=3, n=7 is the main case.
    # If n is larger, this might be slow.
    
    # Simple Greedy
    covered = set()
    selected = []
    remaining = set(all_words)
    
    # Precompute ball for each word
    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.add(v)
        balls.append(ball)
        
    while remaining:
        best_idx = -1
        best_count = -1
        for i, w in enumerate(all_words):
            if w in covered:
                continue
            count = len(balls[i] & remaining)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(all_words[best_idx])
        covered.update(balls[best_idx])
        remaining -= set(balls[best_idx])
        
        if time.time() > deadline - 5.0:
            break
            
    return [list(w) for w in selected]
# EVOLVE-BLOCK-END
