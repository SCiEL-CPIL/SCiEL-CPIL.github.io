import time
import math

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Parameters
    M_MAX = 4000  # Max value in U
    MAX_STEPS = 2000
    
    # Bitsets
    # S_U: bit i is 1 if i in U
    S_U = 1  # 0 is in U
    # S_U_rev: bit (M_MAX - i) is 1 if i in U
    S_U_rev = 1 << M_MAX
    # S_sum: bit i is 1 if i in U+U
    S_sum = 1  # 0+0=0
    # S_diff: bit i is 1 if (i - M_MAX) in U-U
    S_diff = 1 << M_MAX  # 0-0=0 -> index M_MAX
    
    U = [0]
    max_u = 0
    best_score = -1.0
    best_U = [0]
    
    def calc_score(S_sum, S_diff, max_u):
        # Count bits
        # S_sum contains sums 0..2*max_u
        # S_diff contains diffs -max_u..max_u
        # We need |U+U| and |U-U|
        # Note: S_sum might have bits set beyond 2*max_u if we didn't mask?
        # We only add x+u where x,u <= max_u, so sums <= 2*max_u.
        # Diffs are in [-max_u, max_u].
        # We need to count only valid bits?
        # S_sum is built from valid sums.
        # S_diff is built from valid diffs.
        # However, S_diff might have bits outside [0, 2*M_MAX]?
        # Shifts:
        # S_U_rev << x: max bit index M_MAX - 0 + x = M_MAX + x.
        # S_U << (M_MAX - x): max bit index M_MAX + (M_MAX - x) = 2*M_MAX - x.
        # Min index: 0.
        # So S_diff bits are in [0, 2*M_MAX].
        # Valid diffs are -max_u .. max_u.
        # Index range: M_MAX - max_u .. M_MAX + max_u.
        # We should mask S_diff to this range before counting?
        # Or just count all bits?
        # If we only add valid elements, all bits in S_diff correspond to valid diffs?
        # Yes, because S_U only has bits for u in U.
        # So S_diff only has bits for d = u1 - u2.
        # So bit count is exactly |U-U|.
        
        n_sum = bin(S_sum).count('1')
        n_diff = bin(S_diff).count('1')
        
        if n_sum == 0:
            return 0.0
            
        # max_u is the actual max element in U
        range_val = 2 * max_u + 1
        if range_val <= 1:
            return 1.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        log_ratio = math.log(ratio)
        log_range = math.log(range_val)
        
        return 1.0 + log_ratio / log_range

    current_score = calc_score(S_sum, S_diff, 0)
    
    for step in range(MAX_STEPS):
        if time.time() > deadline:
            break
            
        best_candidate = -1
        best_candidate_score = current_score
        best_new_S_U = S_U
        best_new_S_U_rev = S_U_rev
        best_new_S_sum = S_sum
        best_new_S_diff = S_diff
        best_new_max_u = max_u
        
        # Try adding each x
        for x in range(1, M_MAX + 1):
            if (S_U >> x) & 1:
                continue
                
            # Calculate new bitsets
            # S_U_new
            new_S_U = S_U | (1 << x)
            new_S_U_rev = S_U_rev | (1 << (M_MAX - x))
            
            # S_sum_new
            # New sums: x + u for u in U_old, and 2x
            # S_U_old << x gives bits at u+x.
            # We need to OR with 1 << (2x)
            new_S_sum = S_sum | (S_U << x) | (1 << (2 * x))
            
            # S_diff_new
            # New diffs: x - u and u - x
            # Part 1: x - u. Index: M_MAX + x - u.
            # S_U_rev has bit M_MAX - u.
            # Shift left by x: bit M_MAX - u + x. Correct.
            part1 = S_U_rev << x
            
            # Part 2: u - x. Index: M_MAX + u - x.
            # S_U has bit u.
            # Shift left by M_MAX - x: bit u + M_MAX - x. Correct.
            part2 = S_U << (M_MAX - x)
            
            new_S_diff = S_diff | part1 | part2
            
            new_max_u = max(max_u, x)
            
            # Masking?
            # S_sum bits are in 0..2*new_max_u.
            # S_diff bits are in M_MAX - new_max_u .. M_MAX + new_max_u.
            # We don't strictly need to mask for counting if we trust the generation,
            # but S_diff might have old bits? No, we only OR.
            # However, S_diff bits outside the valid range for current max_u?
            # If we added a small x, max_u might not change.
            # If max_u increases, the valid range expands.
            # The bits generated are always within the new valid range.
            # Old bits are within old valid range, which is subset of new valid range.
            # So no invalid bits are introduced.
            
            score = calc_score(new_S_sum, new_S_diff, new_max_u)
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = x
                best_new_S_U = new_S_U
                best_new_S_U_rev = new_S_U_rev
                best_new_S_sum = new_S_sum
                best_new_S_diff = new_S_diff
                best_new_max_u = new_max_u
        
        if best_candidate == -1:
            break
            
        # Accept
        S_U = best_new_S_U
        S_U_rev = best_new_S_U_rev
        S_sum = best_new_S_sum
        S_diff = best_new_S_diff
        max_u = best_new_max_u
        U.append(best_candidate)
        current_score = best_candidate_score
        
        if current_score > best_score:
            best_score = current_score
            best_U = U.copy()
            
        # Optional: try removals? 
        # For now, greedy addition is a strong baseline.
        
    # If we used all time, return best_U
    # Ensure 0 is in best_U
    if 0 not in best_U:
        best_U.append(0)
        
    return sorted(best_U)

