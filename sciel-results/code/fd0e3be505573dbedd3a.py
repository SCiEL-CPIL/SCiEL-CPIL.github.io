import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for the 150s limit
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls) as bitmasks
    # For q=3, n=8, total_words=6561.
    # Bitmask size is manageable in Python ints.
    balls = [0] * total_words
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    # Generate balls
    # Optimization: Only generate balls for the specific radius.
    # For radius 1, ball size is 1 + n*(q-1) = 17.
    # We can precompute the "delta" offsets for each position.
    
    if radius == 1:
        # Precompute all possible single-position changes
        # offsets[pos] = list of (delta, new_idx_offset) ? 
        # Actually, just iterate positions and new digits.
        for i in range(total_words):
            d = digits_of[i]
            mask = 1 << i
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
            balls[i] = mask
    else:
        # General case (slower, but task is r=1)
        import itertools
        for i in range(total_words):
            d = digits_of[i]
            mask = 1 << i
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * q_powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
            balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Strategy: Multi-start Greedy + Local Search
    
    # 1. Greedy Set Cover
    def greedy_cover():
        uncovered = all_mask
        code = []
        # Precompute bit counts for speed? 
        # Greedy step: pick c that maximizes |Ball(c) & uncovered|
        
        # Optimization: 
        # Instead of checking all N words every time (O(N^2 * bit_ops)),
        # we can use a heap or just scan. N=6561 is small enough for a few iterations.
        # Number of codewords ~ 500. 500 * 6561 = 3.2M operations. 
        # Each operation is a bit AND and count. 
        # Python bit_count is fast (C implementation).
        
        while uncovered:
            best_idx = -1
            max_gain = -1
            
            # Optimization: If we have a large number of candidates, 
            # we might want to restrict the search space, but for now full scan.
            for i in range(total_words):
                if not (uncovered >> i) & 1:
                    # If i is already covered, it might still be a good center?
                    # No, if i is covered, adding it doesn't cover i.
                    # But it might cover others.
                    # Actually, standard greedy picks any i.
                    # We can skip if i is already in code, but we don't track that easily here.
                    # Let's just compute gain.
                    pass
                
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if best_idx == -1 or max_gain == 0:
                break
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # 2. Local Search: Delete redundant, then try swaps
    def local_search(code):
        code_set = set(code)
        cover_mask = 0
        for c in code:
            cover_mask |= balls[c]
            
        # Verify full coverage
        if cover_mask != all_mask:
            # Should not happen if greedy was correct, but fix just in case
            uncovered = all_mask & ~cover_mask
            while uncovered:
                lb = uncovered & (-uncovered)
                idx = lb.bit_length() - 1
                uncovered ^= lb
                code_set.add(idx)
                cover_mask |= balls[idx]
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Delete redundant codewords
            # A codeword c is redundant if removing it doesn't uncover any points.
            # i.e., cover_mask_without_c == cover_mask
            # cover_mask_without_c = cover_mask ^ balls[c] is NOT correct because other codewords might cover those points.
            # Correct check: 
            # Let S = {c' in code, c' != c}.
            # Union(S) must cover Ball(c).
            # This is expensive to check directly.
            # Alternative: 
            # For each c in code, check if every point in Ball(c) is covered by some other codeword.
            # This is O(|code| * ball_size * |code|) worst case.
            # Better: 
            # Maintain a count of how many codewords cover each point.
            # counts[w] = number of c in code such that w in Ball(c).
            # A codeword c is redundant if for all w in Ball(c), counts[w] > 1.
            
            # Let's use the count array approach. It's O(N) to build, O(|code|*ball_size) to check.
            
            counts = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                # Add balls[c] to counts
                # Since balls[c] is a bitmask, we need to iterate bits or use a precomputed list of neighbors.
                # Iterating bits of a 6561-bit int is fast in Python.
                m = balls[c]
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    m ^= lb
                    counts[idx] += 1
            
            # Now check for redundancy
            to_remove = []
            for c in code_set:
                m = balls[c]
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    m ^= lb
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            for c in to_remove:
                code_set.remove(c)
                # Update counts
                m = balls[c]
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    m ^= lb
                    counts[idx] -= 1
                improved = True
            
            # Phase 2: Try to replace one codeword with another to allow further deletion?
            # Or just try to swap c with c' if c' covers more "critical" points?
            # A simpler, effective heuristic: 
            # If we are stuck, try removing a codeword c and see if we can cover the newly uncovered points
            # by adding ONE other codeword c' that covers most of them.
            # This is a "1-exchange" move.
            
            if not improved:
                # Try 1-exchange
                # For each c in code_set:
                #   Remove c -> uncovered = Ball(c) \ Union(others)
                #   Find c' not in code_set that covers max(uncovered)
                #   If c' covers ALL uncovered, replace c with c'.
                #   This doesn't reduce size, but might enable a subsequent deletion.
                
                # This might be slow. Let's try a simpler "random restart" or "kick" if stuck.
                # Or just stop if no deletion.
                pass
                
        return list(code_set)

    # 3. Main Loop: Multiple Greedy Starts with Randomization
    
    best_code = None
    best_size = total_words + 1
    
    # We can run multiple greedy starts.
    # Standard greedy is deterministic. 
    # To get diversity, we can randomize the tie-breaking in greedy.
    # Or, we can perturb the "weights" of the balls.
    
    # Let's do a few deterministic runs with different random seeds for tie-breaking.
    
    num_starts = 10
    # Adjust num_starts based on time? 
    # Greedy takes ~0.1s? Local search ~0.1s?
    # 10 starts * 0.2s = 2s. Very safe.
    
    # Actually, let's implement a randomized greedy:
    # When multiple candidates have the same max gain, pick one randomly.
    # Or, with small probability, pick a slightly sub-optimal candidate to escape local minima.
    
    for run in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        uncovered = all_mask
        code = []
        
        # Randomize tie-breaking
        random.seed(run * 12345 + 67890)
        
        while uncovered:
            best_gain = -1
            candidates = []
            
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates or best_gain == 0:
                break
                
            # Pick randomly from best candidates
            c = random.choice(candidates)
            code.append(c)
            uncovered &= ~balls[c]
            
        # Local Search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we hit the known optimum (486), we can stop early?
            # The task says "best known is 486". If we find 486, it's optimal (likely).
            # But we should still try to find 485? No, lower bound is 403.
            # If we find 486, it's a very strong result.
            # Let's continue to see if we can go lower, but 486 is the target.
            if best_size <= 486:
                # Keep searching for a bit longer to see if 485 is possible?
                # Or just break to save time for safety?
                # Given the difficulty, 486 is likely the best we can get quickly.
                # Let's do a few more runs to confirm stability.
                pass

    # Fallback: If no code found (shouldn't happen), return all words? 
    # Or just the greedy result.
    if best_code is None:
        best_code = greedy_cover()
        best_code = local_search(best_code)
        
    # Convert to required format
    return [list(digits_of[i]) for i in best_code]

# Helper for quick testing or fallback if local_search fails
def _greedy_fallback(q, n, radius, balls, digits_of, total_words):
    # This is not used in the main construct above, but kept for signature compatibility if needed
    # The main construct has its own internal greedy.
    pass
