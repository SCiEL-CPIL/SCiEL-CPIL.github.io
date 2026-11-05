import numpy as np
import time
import random

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve(deadline)
    else:
        return _generic(q, n, radius, deadline)

def _solve(deadline):
    q, n, radius = 3, 8, 1
    total = q ** n  # 6561
    
    # Precompute digits for all indices
    # digits[i, j] is the j-th digit of index i
    digits = np.zeros((total, n), dtype=np.int64)
    for j in range(n):
        base = q ** (n - 1 - j)
        digits[:, j] = (np.arange(total) // base) % q
        
    # Precompute balls (indices)
    # ball[i] contains indices of words within distance 1 of i
    balls = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
    for i in range(total):
        d = digits[i]
        balls[i, 0] = i
        idx = 1
        for j in range(n):
            for v in range(q):
                if v != d[j]:
                    # new index = i - d[j]*base + v*base
                    base = q ** (n - 1 - j)
                    balls[i, idx] = i + (v - d[j]) * base
                    idx += 1
                    
    # Precompute boolean masks for balls
    # ball_mask[i, k] is True if k is in ball(i)
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, balls[i]] = True
        
    # Greedy initialization
    uncovered = np.ones(total, dtype=bool)
    code = []
    
    while uncovered.any():
        if time.time() > deadline - 10:
            break
            
        # Calculate gain for each candidate
        # gain[i] = number of uncovered words in ball(i)
        # Using matrix multiplication: ball_mask (bool) dot uncovered (bool)
        # Note: bool dot bool works in numpy? 
        # np.dot(bool, bool) returns int.
        gains = ball_mask.dot(uncovered)
        
        # Select best candidate
        best = np.argmax(gains)
        code.append(best)
        uncovered[balls[best]] = False
        
    code = np.array(code, dtype=np.int64)
    
    # Compute cover_count
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code:
        cover_count[balls[c]] += 1
        
    def removal_pass(code, cover_count):
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if removable
                if cover_count[balls[c]].min() >= 2:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_code.append(c)
            code = np.array(new_code, dtype=np.int64) if new_code else np.array([], dtype=np.int64)
        return code
        
    code = removal_pass(code, cover_count)
    
    # Local Search: Swaps
    random.seed(42)
    last_improvement = time.time()
    
    while time.time() < deadline - 5:
        if len(code) == 0:
            break
            
        # Pick random codeword to replace
        i = random.randrange(len(code))
        c_old = code[i]
        
        # Pick random new codeword
        c_new = random.randrange(total)
        
        if c_new == c_old:
            continue
            
        # Check validity of swap c_old -> c_new
        # Words that lose coverage: Ball(c_old) \ Ball(c_new)
        # We need cover_count[w] >= 2 for all w in that set
        
        # Efficient check using boolean masks
        # diff_mask = ball_mask[c_old] & ~ball_mask[c_new]
        # if np.any(cover_count[diff_mask] < 2): invalid
        
        # To avoid creating diff_mask explicitly (memory), we can do:
        # in_old = ball_mask[c_old]
        # in_new = ball_mask[c_new]
        # # We need to check words where in_old is True and in_new is False
        # # cover_count[in_old & ~in_new].min() >= 2
        
        # Optimization:
        # If c_new is in code, and c_new != c_old, we might have duplicates? 
        # The problem says codewords must be distinct. 
        # If c_new is already in code, adding it creates a duplicate. 
        # We should avoid duplicates or handle them.
        # Greedy ensures distinct. Swaps might introduce duplicates if c_new is in code.
        # If c_new is in code, then we are effectively moving c_old to c_new's position, 
        # but c_new is already there. This reduces code size by 1 if we just remove c_old?
        # No, if c_new is in code, and we replace c_old with c_new, we have two c_new.
        # We should check if c_new is in code.
        
        if c_new in code:
            # If c_new is already in the code, we can just try to remove c_old
            # because c_new is already covering everything c_new covers.
            # But wait, if c_new is in code, then c_new's ball is already counted in cover_count.
            # If we remove c_old, we lose c_old's ball.
            # This is equivalent to a removal attempt.
            # Let's just treat it as a removal if c_new is in code.
            if cover_count[balls[c_old]].min() >= 2:
                cover_count[balls[c_old]] -= 1
                code = np.delete(code, i)
                # Run removal pass again to clean up
                code = removal_pass(code, cover_count)
                last_improvement = time.time()
            continue
            
        # Check swap validity
        # diff = ball_mask[c_old] & ~ball_mask[c_new]
        # We can compute this without explicit boolean array allocation if we use indexing?
        # No, boolean indexing is fast.
        
        # Create the mask
        # in_old = ball_mask[c_old]
        # in_new = ball_mask[c_new]
        # diff_indices = np.where(in_old & ~in_new)[0]
        # if diff_indices.size > 0:
        #     if np.any(cover_count[diff_indices] < 2):
        #         continue
        
        # Faster:
        # min_cov = np.min(cover_count[ball_mask[c_old] & ~ball_mask[c_new]])
        # But creating the boolean array is O(total).
        
        in_old = ball_mask[c_old]
        in_new = ball_mask[c_new]
        # Check if any word in in_old but not in_in_new has cover < 2
        # We can do:
        # potential_bad = in_old & ~in_new
        # if np.any(cover_count[potential_bad] < 2):
        #     continue
        
        # To save memory/time, we can check:
        # min_val = np.min(cover_count[in_old & ~in_new])
        
        # Let's just do it. 6561 bytes is small.
        if np.any(cover_count[in_old & ~in_new] < 2):
            continue
            
        # Swap is valid
        # Apply swap
        cover_count[balls[c_old]] -= 1
        cover_count[balls[c_new]] += 1
        code[i] = c_new
        
        # Try to remove
        old_len = len(code)
        code = removal_pass(code, cover_count)
        
        if len(code) < old_len:
            last_improvement = time.time()
        else:
            # Revert
            cover_count[balls[c_new]] -= 1
            cover_count[balls[c_old]] += 1
            code[i] = c_old
            # Note: removal_pass didn't change anything, so cover_count is consistent with code
            # after revert? 
            # In removal_pass, if no removal happened, cover_count is unchanged.
            # So revert is safe.
            
    # Convert to result format
    result = []
    for idx in code:
        word = digits[idx].tolist()
        result.append(word)
        
    return result

def _generic(q, n, radius, deadline):
    # Fallback for other parameters (not expected in this task)
    # Simple greedy
    total = q ** n
    digits = np.zeros((total, n), dtype=np.int64)
    for j in range(n):
        base = q ** (n - 1 - j)
        digits[:, j] = (np.arange(total) // base) % q
        
    # Precompute balls
    balls = []
    for i in range(total):
        d = digits[i]
        ball = [i]
        for j in range(n):
            for v in range(q):
                if v != d[j]:
                    base = q ** (n - 1 - j)
                    ball.append(i + (v - d[j]) * base)
        balls.append(ball)
        
    uncovered = np.ones(total, dtype=bool)
    code = []
    while uncovered.any():
        if time.time() > deadline - 5:
            break
        gains = []
        for c in range(total):
            gains.append(sum(1 for w in balls[c] if uncovered[w]))
        best = np.argmax(gains)
        code.append(best)
        for w in balls[best]:
            uncovered[w] = False
            
    result = []
    for idx in code:
        result.append(digits[idx].tolist())
    return result
