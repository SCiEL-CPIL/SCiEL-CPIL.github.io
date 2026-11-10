import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        # coeffs is a numpy array
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        # Check constraints
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        if not validate(np.array(poly_list)):
            return
        score = evaluate(np.array(poly_list))
        if score < best_score:
            best_score = score
            best_poly = np.array(poly_list, dtype=np.int64).copy()

    # Phase 1: Structured "Two-Block" Search
    # Hypothesis: P(x) = Block1(x) - Block2(x) where blocks are contiguous and have uniform sign
    # This creates cancellations in the cross terms.
    
    # We can also allow the blocks to have different internal patterns, 
    # but let's start with uniform +/-1.
    
    # L1 + L2 = 12
    for L1 in range(1, 12):
        L2 = 12 - L1
        # S1 is start of block 1
        # S2 is start of block 2
        # To avoid duplicates and ensure valid range, 
        # we can fix S1=0 by shifting? 
        # Shifting P(x) by x^k shifts P^2 by x^{2k}. Sparsity is invariant.
        # So we can fix the minimum exponent of P to be 0.
        # This means min(S1, S2) must be 0? 
        # Not necessarily, if we have gaps.
        # But for contiguous blocks, if we shift so that min exponent is 0,
        # then either S1=0 or S2=0 (if they are the leftmost block).
        
        # Case A: Block 1 is leftmost (S1 < S2)
        # Shift so S1 = 0.
        # S2 can be anything such that S2 + L2 - 1 <= 40.
        # Also S2 > S1 + L1 - 1 ? No, they can overlap? 
        # If they overlap, it's not "two blocks of opposite signs" in a simple disjoint sense.
        # The finding said "two blocks of opposite signs". Usually implies disjoint or distinct regions.
        # Let's allow disjoint blocks first.
        
        # If S1=0, then Block 1 occupies [0, L1-1].
        # Block 2 occupies [S2, S2+L2-1].
        # For them to be disjoint and Block 2 to be to the right: S2 >= L1.
        # Max degree: S2 + L2 - 1 <= 40.
        
        for S2 in range(L1, 41 - L2 + 1):
            # Construct P
            c = [0] * 41
            for i in range(L1):
                c[i] = 1
            for i in range(L2):
                c[S2 + i] = -1
            
            # Ensure last coeff is nonzero (it is -1)
            # Ensure length <= 41 (it is)
            # Ensure 12 nonzero terms (it is)
            try_poly(c)
            
            # Also try flipped signs: Block 1 is -1, Block 2 is 1
            # This is equivalent to -P(x), which has same square. 
            # So we don't need to check both.
            
        # Case B: Block 2 is leftmost (S2 < S1)
        # Shift so S2 = 0.
        # Block 2 occupies [0, L2-1].
        # Block 1 occupies [S1, S1+L1-1].
        # Disjoint: S1 >= L2.
        # Max degree: S1 + L1 - 1 <= 40.
        
        for S1 in range(L2, 41 - L1 + 1):
            c = [0] * 41
            for i in range(L2):
                c[i] = -1 # Block 2 (leftmost) is negative
            for i in range(L1):
                c[S1 + i] = 1  # Block 1 (rightmost) is positive
                
            try_poly(c)
            
        # Case C: Overlapping blocks?
        # If they overlap, the coefficients add up.
        # e.g. 1 - 1 = 0. This reduces the number of terms!
        # We need exactly 12 non-zero terms in P.
        # If we start with two blocks of 12 terms total, and they overlap,
        # the resulting polynomial might have fewer than 12 terms.
        # The constraint is "exactly 12 nonzero terms".
        # So we should check if the generated poly has 12 terms.
        # My try_poly does this.
        
        # Let's allow overlapping blocks in the search space.
        # S1=0. S2 can be from 0 to 40-L2+1.
        # We already covered S2 >= L1 (disjoint right).
        # Now cover S2 < L1 (overlap or left).
        # If S2 < L1, they overlap or Block 2 is to the left.
        # But we fixed S1=0. So Block 2 cannot be to the left (S2 >= 0).
        # So S2 is in [0, L1-1] for overlap.
        
        for S2 in range(0, L1):
            c = [0] * 41
            for i in range(L1):
                c[i] = 1
            for i in range(L2):
                c[S2 + i] -= 1 # Subtract block 2
            
            # Check if it has exactly 12 non-zeros
            if np.count_nonzero(c) == 12:
                try_poly(c)
                
            # Try flipped signs
            c2 = [-x for x in c]
            if np.count_nonzero(c2) == 12:
                try_poly(c2)
                
    # Phase 2: Backtracking CSP for general supports with +/-1 coefficients
    # We use the best score found so far as a bound.
    
    # State: list of (exponent, coeff)
    # Start with (0, 1)
    
    def backtrack(current_terms, current_square, current_nonzero_count):
        nonlocal best_score, best_poly, deadline
        
        if time.time() > deadline - 5:
            return
            
        if len(current_terms) == 12:
            # Check if last coeff is nonzero (it is, we add non-zero)
            # Construct full poly
            c = [0] * 41
            for e, val in current_terms:
                c[e] = val
            try_poly(c)
            return

        # Pruning: if current_nonzero_count > best_score, no point continuing
        # Note: adding more terms can only increase or maintain the number of non-zeros in the square?
        # No, adding a term can create cancellations!
        # So we cannot prune based on current count > best_score.
        # We CAN prune if current_nonzero_count + (min_possible_increase) > best_score?
        # Hard to estimate.
        # However, if current_nonzero_count is already very high, it's unlikely to drop.
        # Let's use a softer bound: if current_nonzero_count > best_score + 5, prune?
        # Or just rely on time.
        
        # Heuristic: Try to place the next term such that it cancels existing terms.
        # Existing square S.
        # New term (e, s) with s in {-1, 1}.
        # New square S' = S + s^2 x^{2e} + 2s sum_{(e_i, v_i) in current_terms} v_i x^{e+e_i}.
        
        last_e = current_terms[-1][0] if current_terms else -1
        
        # Try exponents from last_e + 1 to 40
        # To speed up, we can prioritize exponents that might cause cancellations.
        # Cancellations happen if the new cross terms match existing non-zero terms with opposite signs.
        
        # For simplicity, iterate through all possible next exponents.
        # Branching factor is up to 40. Depth 11. 40^11 is too big.
        # We need strong pruning or heuristic ordering.
        
        # Let's try a limited search: only consider exponents that are "close" to existing ones 
        # or create specific patterns? 
        # Actually, let's just do a random subset of backtracking or a very shallow one.
        
        # Alternative: Use the "Two-Block" result as a seed for a local search that allows 
        # moving terms (changing exponents) and changing coefficients.
        
        pass # Skipping full backtracking due to complexity, relying on Phase 1 and Phase 3.

    # Phase 3: Local Search on the best candidate from Phase 1
    if best_poly is not None:
        curr = best_poly.copy()
        curr_score = best_score
        no_improve = 0
        max_no_improve = 5000
        temp = 1.0
        random.seed(42)

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Randomly modify one coefficient
            idx = random.randint(0, len(curr)-1)
            old_val = curr[idx]
            
            # Try changing value
            if random.random() < 0.7:
                new_val = random.randint(-8, 8)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
            else:
                # Try moving the term (set old to 0, set new to old_val)
                # This changes the support
                new_idx = random.randint(0, len(curr)-1)
                if new_idx == idx:
                    continue
                if curr[new_idx] != 0:
                    # Can't move to occupied slot easily in this representation without shifting
                    # Just change value for now
                    new_val = random.randint(-8, 8)
                    if new_val == 0:
                        new_val = 1 if random.random() < 0.5 else -1
                else:
                    curr[idx] = 0
                    curr[new_idx] = old_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            if best_score <= 7:
                                break
                    else:
                        delta = new_score - curr_score
                        if random.random() < np.exp(-delta / max(temp, 0.1)):
                            curr_score = new_score
                            no_improve = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr.copy()
                        else:
                            curr[new_idx] = 0
                            curr[idx] = old_val
                            no_improve += 1
                    temp *= 0.9995
                    continue

            if new_val == old_val:
                continue

            curr[idx] = new_val
            new_score = evaluate(curr)

            if new_score < curr_score:
                curr_score = new_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
                    if best_score <= 7:
                        break
            else:
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temp, 0.1)):
                    curr_score = new_score
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                else:
                    curr[idx] = old_val
                    no_improve += 1

            temp *= 0.9995

    # Ensure final coefficient is nonzero
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
