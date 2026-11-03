import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    # Parameters for the specific problem K_3(6,1)
    # q=3, n=6, radius=1
    # Total words = 729
    
    # Helper functions
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute balls as bitmasks
    # Each ball is an integer with bits set for all words in distance <= radius
    all_words = list(range(q**n))
    balls = [0] * len(all_words)
    
    # Precompute neighbors for each word to build balls
    # dist 1 neighbors
    for w in all_words:
        w_list = int_to_list(w, n)
        mask = 1 << w
        balls[w] = mask
        for i in range(n):
            orig = w_list[i]
            for d in range(q):
                if d != orig:
                    new_word = w_list.copy()
                    new_word[i] = d
                    nw = list_to_int(new_word)
                    balls[w] |= (1 << nw)
                    
    # Greedy Set Cover Initialization
    code = []
    covered_mask = 0
    full_mask = (1 << len(all_words)) - 1
    cover_count = np.zeros(len(all_words), dtype=np.int8)
    
    # Greedy phase
    while covered_mask != full_mask:
        best_w = -1
        best_gain = -1
        # To speed up, we can sample or iterate. 729 is small, iterate all.
        for w in all_words:
            # Gain = number of new bits covered by ball[w]
            gain = bin(balls[w] & ~covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_w = w
                if gain == 0:
                    break # Should not happen if not full
        
        if best_w == -1:
            break
            
        code.append(best_w)
        covered_mask |= balls[best_w]
        # Update cover counts
        # Iterate bits in ball[best_w]
        mask = balls[best_w]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask ^= lsb
            
    initial_size = len(code)
    
    # Simulated Annealing / Local Search
    # State: code (list of ints), covered_mask, cover_count
    
    def is_removable(c):
        # Check if min cover_count in ball[c] is >= 2
        mask = balls[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if cover_count[idx] < 2:
                return False
            mask ^= lsb
        return True

    def try_remove(c):
        nonlocal covered_mask
        if is_removable(c):
            code.remove(c)
            # Update counts
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cover_count[idx] -= 1
                mask ^= lsb
            # Update mask
            # We don't easily update covered_mask by subtraction with OR.
            # But we can recalculate or just rely on cover_count for validity?
            # For validity check in swaps, we need to know if a word is covered by *others*.
            # cover_count > 0 means covered.
            # So we can just use cover_count.
            return True
        return False

    def try_swap(c, w_new):
        # Replace c with w_new
        # Condition: All words in B(c) \ B(w_new) must have cover_count >= 2
        # (So that removing c doesn't uncover them)
        # Words in B(w_new) \ B(c) will have cover_count increased, which is fine.
        
        # Identify words in B(c) that are NOT in B(w_new)
        # mask_diff = balls[c] & ~balls[w_new]
        mask_diff = balls[c] & ~balls[w_new]
        
        while mask_diff:
            lsb = mask_diff & -mask_diff
            idx = lsb.bit_length() - 1
            if cover_count[idx] < 2:
                return False
            mask_diff ^= lsb
            
        # If valid, perform swap
        code[code.index(c)] = w_new
        
        # Update counts
        # Remove c
        mask_c = balls[c]
        while mask_c:
            lsb = mask_c & -mask_c
            idx = lsb.bit_length() - 1
            cover_count[idx] -= 1
            mask_c ^= lsb
            
        # Add w_new
        mask_new = balls[w_new]
        while mask_new:
            lsb = mask_new & -mask_new
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask_new ^= lsb
            
        return True

    # Main SA Loop
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    best_code = list(code)
    best_size = len(code)
    
    # Precompute neighbors for swaps (distance 1 and 2) to speed up candidate generation
    # For each c, neighbors are words in B(c) \ {c}
    # We can just pick a random word from the ball or nearby.
    
    while time.time() < deadline - 10 and T > T_min:
        # 1. Try to remove
        if code:
            c = random.choice(code)
            if is_removable(c):
                # Always accept removal
                try_remove(c)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue

        # 2. Try to swap
        if code:
            c = random.choice(code)
            # Pick a random neighbor from B(c)
            # Sample a random word from the ball
            # We can pick a random offset
            # Simpler: pick a random word from all_words? No, too sparse.
            # Pick from B(c)
            # Get list of words in B(c)? Too slow to build list each time.
            # We can precompute lists of words in each ball.
            
            # Let's precompute ball_lists
            if not hasattr(construct, 'ball_lists'):
                construct.ball_lists = []
                for w in all_words:
                    mask = balls[w]
                    lst = []
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        lst.append(idx)
                        mask ^= lsb
                    construct.ball_lists.append(lst)
            
            neighbors = construct.ball_lists[c]
            if neighbors:
                w_new = random.choice(neighbors)
                if w_new != c:
                    # Try swap
                    # Check validity
                    mask_diff = balls[c] & ~balls[w_new]
                    valid = True
                    while mask_diff:
                        lsb = mask_diff & -mask_diff
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] < 2:
                            valid = False
                            break
                        mask_diff ^= lsb
                    
                    if valid:
                        # Accept swap (delta E = 0)
                        # Perform swap logic
                        idx_c = code.index(c)
                        code[idx_c] = w_new
                        
                        mask_c = balls[c]
                        while mask_c:
                            lsb = mask_c & -mask_c
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            mask_c ^= lsb
                            
                        mask_new = balls[w_new]
                        while mask_new:
                            lsb = mask_new & -mask_new
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            mask_new ^= lsb
                        continue

        # 3. Random perturbation (Add then Remove?)
        # Sometimes we need to add a word to enable a better configuration?
        # Or just pick a random word to move a codeword to?
        
        # If no move accepted, cool down
        T *= cooling

    # Restore best
    code = best_code
    
    # Final polish: Try to remove any remaining removable codewords
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for c in list(code):
            if is_removable(c):
                try_remove(c)
                changed = True

    result = [int_to_list(c, n) for c in code]
    return result
