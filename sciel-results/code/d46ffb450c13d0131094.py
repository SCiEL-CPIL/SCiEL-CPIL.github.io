import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if int(np.count_nonzero(coeffs)) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        arr = np.array(poly_list, dtype=np.int64)
        # Trim trailing zeros for validation, but keep length for internal use if needed?
        # The problem says "Return coefficient list in ascending degree, length<=41, nonzero final coefficient"
        # So we should return a trimmed list.
        # However, internally we work with a fixed length array of 12 non-zeros?
        # No, the polynomial has exactly 12 non-zero terms.
        # The representation is a list of coefficients.
        # If I have [1, 0, 1, 0, ...], the non-zero count is 2.
        # The constraint is "exactly 12 nonzero terms" in the INPUT polynomial.
        
        # Check if exactly 12 non-zero terms
        if int(np.count_nonzero(arr)) != 12:
            return
        if not np.all((arr >= -64) & (arr <= 64)):
            return
        if arr[-1] == 0:
            return
            
        score = evaluate(arr)
        if score < best_score:
            best_score = score
            best_poly = arr.copy()

    random.seed(42)

    # Phase 1: Generate diverse initial candidates
    # 1. Random sparse polynomials
    # 2. Structured polynomials (e.g. based on known sparse squares)
    
    candidates = []
    
    # Generate random polynomials with 12 non-zero terms
    for _ in range(500):
        # Choose 12 distinct indices
        idxs = random.sample(range(41), 12)
        idxs.sort()
        coeffs = [0] * 41
        for i in idxs:
            # Random coefficient
            # Bias towards small integers for better cancellation potential?
            coeffs[i] = random.choice([-2, -1, 1, 2, -3, 3])
        candidates.append(coeffs)

    # Try known "good" patterns
    # e.g. 1 + 2x - 2x^2 type patterns scaled/shifted
    base_patterns = [
        [1, 2, -2],
        [1, -1, 1],
        [1, 1, -2, 1],
        [1, -2, 2, -1]
    ]
    
    for pat in base_patterns:
        # Pad to 12 terms by adding zeros? No, must be exactly 12 non-zero.
        # So we need to add more terms.
        # Just use the random ones for now, but seed with some structure.
        pass

    # Evaluate all candidates
    for c in candidates:
        try_poly(c)
        if time.time() > deadline - 5:
            break

    if best_poly is None:
        # Fallback
        best_poly = np.array([1]*12 + [0]*29, dtype=np.int64)
        best_score = evaluate(best_poly)

    # Phase 2: Aggressive Local Search
    # We will run local search on the best found and on some other good candidates.
    # To save time, we focus on the global best.
    
    curr = best_poly.copy()
    curr_score = best_score
    no_improve = 0
    max_no_improve = 20000 # Allow more steps
    
    # Value range for mutations
    # Small values are more likely to create exact cancellations.
    # But larger values might be needed.
    # We'll sample from [-64, 64] but with a bias towards small numbers.
    
    def get_random_val():
        r = random.random()
        if r < 0.5:
            return random.choice([-1, 1])
        elif r < 0.8:
            return random.choice([-2, 2, -3, 3])
        elif r < 0.95:
            return random.choice([-4, 4, -5, 5, -10, 10])
        else:
            return random.randint(-64, 64)

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Pick a random index to modify
        # We should only modify indices that are currently non-zero?
        # Or can we move the non-zero terms?
        # The constraint is "exactly 12 nonzero terms".
        # If I change a non-zero to zero, I have 11. I must change a zero to non-zero.
        # This is a "move" operation.
        # Alternatively, I can just change the value of an existing non-zero term.
        # Changing value keeps the count at 12 (unless it becomes 0, then I need to fix).
        
        # Strategy 1: Change value of a non-zero term.
        # Strategy 2: Swap a non-zero term with a zero term (move support).
        
        op = random.random()
        
        if op < 0.8:
            # Change value of a non-zero term
            non_zero_idxs = np.nonzero(curr)[0]
            if len(non_zero_idxs) == 0: break
            idx = random.choice(non_zero_idxs)
            old_val = curr[idx]
            new_val = get_random_val()
            if new_val == 0:
                # If we set it to 0, we violate the "12 non-zero" constraint temporarily.
                # We must set another zero to non-zero.
                # This is complex. Let's avoid 0 in mutation for simplicity in this loop.
                # Or handle it: pick a zero index, set it to a random non-zero.
                zero_idxs = np.nonzero(curr == 0)[0]
                if len(zero_idxs) == 0:
                    new_val = 1 # Fallback
                else:
                    new_zero_idx = random.choice(zero_idxs)
                    new_zero_val = random.choice([-1, 1, -2, 2])
                    curr[idx] = 0
                    curr[new_zero_idx] = new_zero_val
                    # Evaluate
                    if validate(curr):
                        new_score = evaluate(curr)
                        if new_score < curr_score:
                            curr_score = new_score
                            no_improve = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr.copy()
                                if best_score <= 13:
                                    break
                        else:
                            curr[idx] = old_val
                            curr[new_zero_idx] = 0
                            no_improve += 1
                    else:
                        curr[idx] = old_val
                        curr[new_zero_idx] = 0
                        no_improve += 1
            else:
                curr[idx] = new_val
                if validate(curr):
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            if best_score <= 13:
                                break
                    else:
                        curr[idx] = old_val
                        no_improve += 1
                else:
                    curr[idx] = old_val
                    no_improve += 1
        else:
            # Move a non-zero term to a different position
            non_zero_idxs = np.nonzero(curr)[0]
            zero_idxs = np.nonzero(curr == 0)[0]
            if len(non_zero_idxs) == 0 or len(zero_idxs) == 0:
                continue
            src_idx = random.choice(non_zero_idxs)
            dst_idx = random.choice(zero_idxs)
            val = curr[src_idx]
            curr[src_idx] = 0
            curr[dst_idx] = val
            if validate(curr):
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        if best_score <= 13:
                            break
                else:
                    curr[dst_idx] = 0
                    curr[src_idx] = val
                    no_improve += 1
            else:
                curr[dst_idx] = 0
                curr[src_idx] = val
                no_improve += 1

    # Trim trailing zeros
    if best_poly is not None:
        while len(best_poly) > 0 and best_poly[-1] == 0:
            best_poly = best_poly[:-1]
        # Ensure length <= 41
        if len(best_poly) > 41:
            best_poly = best_poly[:41]
            # Re-validate? If we cut, we might have trailing zeros or changed count.
            # But we started with <= 41 and only modified within 0..40.
            # So length is always 41 initially.
            # Trimming is safe.
            
        # Final check
        if validate(best_poly):
            return best_poly.tolist()
    
    # Fallback
    return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
