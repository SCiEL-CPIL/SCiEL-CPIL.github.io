"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187
    p3 = [1, 3, 9, 27, 81, 243, 729]

    # Precompute ball_masks using integer arithmetic
    ball_masks = [0] * N
    for i in range(N):
        d = [i, i, i, i, i, i, i]
        t = i
        for pos in range(7):
            d[pos] = t % 3
            t //= 3
        mask = 1 << i
        for pos in range(7):
            orig = d[pos]
            for val in (orig - 1, orig + 1):
                if 0 <= val < 3:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask

    # Helper to convert indices to digit lists
    def idx_to_digits(idx):
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        return digits

    # Phase 1: Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1
        selected = []
        
        while uncovered:
            best_idx = -1
            best_cnt = -1
            # Use a randomized subset to speed up search in later iterations
            # but ensure we still find good candidates
            # For simplicity and correctness, we scan all, but with random tie-breaking
            # Actually, scanning all 2187 items per step is fast enough.
            # We can optimize by only checking items not already selected? 
            # Not strictly necessary for correctness, but helps avoid re-adding same item.
            
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for i in candidates:
                cnt = (uncovered & ball_masks[i]).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
            if time.time() > deadline - 10.0:
                break
                
        return selected

    # Phase 2: Local Search (Delete/Redundancy removal)
    def local_search_minimize(code, seed, time_limit):
        rng = random.Random(seed)
        code_set = set(code)
        
        # Precompute coverage counts
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            mask = ball_masks[c]
            # Unpack mask to update counts
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
        
        best_code = code.copy()
        best_size = len(code)
        current_code = code.copy()
        current_set = set(current_code)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove any codeword that is not essential
            # Essential if it uniquely covers at least one word
            removable = []
            for c in current_code:
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
                    removable.append(c)
            
            if removable:
                # Remove all removable at once? Or one by one?
                # Removing one by one is safer to maintain count consistency easily
                # But removing all at once is faster. Let's remove all.
                for c in removable:
                    current_code.remove(c)
                    current_set.remove(c)
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code.copy()
                improved = True
            
        return best_code

    # Phase 3: Local Search with Swaps (to escape local minima)
    def local_search_swap(code, seed, time_limit):
        rng = random.Random(seed)
        code_set = set(code)
        
        cnt = np.zeros(N, dtype=np.int32)
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
        current_code = code.copy()
        current_set = set(current_code)
        
        # First pass: remove redundancies
        current_code = local_search_minimize(current_code, seed, time_limit)
        best_code = current_code
        best_size = len(current_code)
        
        # Recompute counts for best_code
        cnt = np.zeros(N, dtype=np.int32)
        for c in best_code:
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
        
        while time.time() < time_limit:
            improved = False
            # Try random swaps
            # Pick a random codeword c
            if not current_code:
                break
                
            c = rng.choice(current_code)
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
                # c is redundant, remove it
                current_code.remove(c)
                current_set.remove(c)
                mask = ball_masks[c]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code.copy()
                improved = True
                continue
            
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
                if idx not in current_set:
                    candidates.append(idx)
                m ^= lsb
            
            if not candidates:
                continue
            
            c_prime = rng.choice(candidates)
            
            # Perform swap
            current_code.remove(c)
            current_set.remove(c)
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] -= 1
                m ^= lsb
            
            current_code.append(c_prime)
            current_set.add(c_prime)
            mask = ball_masks[c_prime]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
            
            # Check if size improved (swap doesn't change size, but might enable future deletions)
            # So we don't update best_size here, but we continue.
            # However, if we swap and then can delete, that's a net improvement.
            # So we should run a mini-deletion pass after a successful swap?
            # Or just let the outer loop handle it.
            improved = True
            
            if not improved:
                # If no swap was made, try perturbation?
                # For now, just continue
                pass
        
        return best_code

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run multiple starts
    seed = 42
    while time.time() < deadline - 5.0:
        init_code = greedy_cover(seed)
        if not init_code:
            break
        
        # Refine
        refined_code = local_search_swap(init_code, seed, deadline - 5.0)
        
        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_solution = refined_code.copy()
            
        seed += 1
        
        # If we hit the theoretical lower bound, stop
        if best_size <= 156:
            break

    if best_solution is None:
        # Fallback
        return []

    result = []
    for idx in best_solution:
        result.append(idx_to_digits(idx))
    
    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback
    all_words = []
    for i in range(q**n):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_words.append(tuple(digits))
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.add(v)
        balls.append(ball)
    
    covered = set()
    selected = []
    remaining = set(all_words)
    
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
