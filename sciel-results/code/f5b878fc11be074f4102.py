import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 105.0
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball offsets
    ball_offsets = [[0]]
    for coord in range(n):
        b = base[coord]
        for v in range(q):
            ball_offsets.append((v) * b)
    ball_offsets = np.array(ball_offsets, dtype=np.int64)
    
    def idx_to_word(idx):
        w = np.zeros(n, dtype=np.int8)
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        return w

    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        
        # Sort indices by some heuristic to maybe speed up? Or just iterate.
        # Iterate is fine for 2187.
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                if covered[c]:
                    # This optimization is dangerous if c covers other things.
                    # Stick to standard greedy.
                    pass
                ball = c + ball_offsets
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[best_c + ball_offsets] = True
        return code

    current_code = greedy_build()
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[c + ball_offsets] += 1

    def delete_redundant(code_list, cnt_arr):
        """Remove redundant codewords. Returns the modified list."""
        changed = True
        while changed:
            changed = False
            i = 0
            new_list = []
            while i < len(code_list):
                c = code_list[i]
                ball = c + ball_offsets
                if np.min(cnt_arr[ball]) >= 2:
                    # Remove
                    cnt_arr[ball] -= 1
                    changed = True
                else:
                    new_list.append(c)
                    i += 1
            if changed:
                code_list = new_list
        return code_list

    current_code = delete_redundant(current_code, cnt)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)

    # Local Search
    # We maintain a set for O(1) lookup if needed, but list is fine.
    # To speed up swap evaluation, we can precompute which pairs are "valid" swaps?
    # Or just evaluate on the fly.
    
    code_list = current_code
    code_set = set(code_list)
    
    def try_swap(remove_idx, add_idx):
        """
        Attempts a swap. Returns True if successful and updates state.
        """
        if add_idx in code_set:
            return False
        if remove_idx == add_idx:
            return False
            
        b_rem = remove_idx + ball_offsets
        b_add = add_idx + ball_offsets
        
        # Check validity:
        # For all x in b_rem, if x not in b_add, then cnt[x] >= 2.
        # Equivalently: min_{x in b_rem \ b_add} cnt[x] >= 2.
        
        # Efficient check:
        # We can check if cnt[intersection] handles the rest? No.
        # We need to check specifically the parts of b_rem not covered by b_add.
        
        # Vectorized approach:
        # Create a mask for b_rem
        # But we don't have a membership matrix.
        
        # Let's use the fact that if cnt[x] < 2 for any x in b_rem, 
        # AND x is not in b_add, then it's invalid.
        
        # Simpler:
        # Check if any x in b_rem has cnt[x] == 1 AND x is not in b_add.
        
        # Since we don't have a fast "is x in b_add" for a vector of x without membership matrix,
        # let's construct one or use a set.
        # Given total=2187, ball_size=14.
        # Converting b_add to a set of 14 items is fast.
        
        add_set = set(b_add.tolist())
        
        valid = True
        for x in b_rem:
            if cnt[x] < 2 and x not in add_set:
                valid = False
                break
                
        if not valid:
            return False
            
        # Perform swap
        code_list.remove(remove_idx)
        code_list.append(add_idx)
        code_set.remove(remove_idx)
        code_set.add(add_idx)
        cnt[b_rem] -= 1
        cnt[b_add] += 1
        return True

    # Simulated Annealing / Random Local Search
    # We will alternate between random swaps and greedy deletion.
    
    temp = 1.0
    cooling = 0.99
    best_in_sa = best_size
    best_code_sa = list(best_code)
    
    # Pre-generate a list of indices to shuffle for random access?
    all_indices = list(range(total))
    
    steps = 0
    while time.time() < deadline - 5.0:
        # Periodically try to delete redundant
        if steps % 50 == 0:
            new_list = delete_redundant(code_list, cnt)
            if len(new_list) < len(code_list):
                # Update state
                # delete_redundant modified cnt internally? No, it took cnt_arr.
                # It modified cnt_arr.
                # But we need to ensure code_list and code_set are updated.
                # My delete_redundant returns the list, but I passed cnt_arr which was modified.
                # Let's just call a function that updates all three.
                pass # Handled below
            
        # Random Move
        if random.random() < 0.5 and len(code_list) > 0:
            # Swap
            r_idx = random.randint(0, len(code_list) - 1)
            c_rem = code_list[r_idx]
            c_add = random.choice(all_indices)
            
            if try_swap(c_rem, c_add):
                # Accept
                # Update best if improved
                current_size = len(code_list)
                if current_size < best_in_sa:
                    best_in_sa = current_size
                    best_code_sa = list(code_list)
                    best_size = current_size
                    best_code = list(code_list)
            # Note: try_swap updates state if successful. If not, state is unchanged.
        else:
            # Attempt a "deletion" move directly?
            # Or just let the periodic deletion handle it.
            # For now, just cool down.
            pass
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.5
            # Re-shuffle?
            pass
        steps += 1
        
    # Final cleanup
    # Ensure we return the best found
    return [list(idx_to_word(idx)) for idx in best_code]

# Wait, the delete_redundant function in the draft above had a bug: it didn't update code_set or code_list correctly in the main loop if it was just a helper.
# And I didn't actually call a working version that updates the main state variables properly in the loop.
# Let's rewrite with a cleaner state management.

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 105.0
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute ball offsets
    ball_offsets = [0]
    for coord in range(n):
        b = base[coord]
        for v in range(q):
            ball_offsets.append(v * b)
    ball_offsets = np.array(ball_offsets, dtype=np.int64)
    
    def idx_to_word(idx):
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        return w

    # Greedy Initialization
    covered = np.zeros(total, dtype=bool)
    code_list = []
    while not np.all(covered):
        best_c = -1
        max_new = -1
        for c in range(total):
            # Optimization: if c is already covered, it might still be a good choice
            # if it covers many uncovered neighbors.
            ball = c + ball_offsets
            new_cov = np.sum(~covered[ball])
            if new_cov > max_new:
                max_new = new_cov
                best_c = c
                if max_new == ball_size:
                    break
        code_list.append(best_c)
        covered[best_c + ball_offsets] = True
        
    # State
    cnt = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cnt[c + ball_offsets] += 1
        
    code_set = set(code_list)
    
    def attempt_deletion():
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        to_remove = []
        for c in code_list:
            ball = c + ball_offsets
            if np.min(cnt[ball]) >= 2:
                to_remove.append(c)
        for c in to_remove:
            code_list.remove(c)
            code_set.remove(c)
            cnt[c + ball_offsets] -= 1
            changed = True
        return changed

    # Initial cleanup
    while attempt_deletion():
        pass
        
    best_size = len(code_list)
    best_code = list(code_list)
    
    # Local Search with Swaps
    temp = 1.0
    cooling = 0.995
    all_indices = list(range(total))
    
    while time.time() < deadline - 2.0:
        # Try to delete
        if random.random() < 0.1:
            if attempt_deletion():
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                    # Reheat?
                    temp = max(temp, 0.5)
                continue

        # Try Swap
        if len(code_list) > 0:
            r_idx = random.randint(0, len(code_list) - 1)
            c_rem = code_list[r_idx]
            c_add = random.choice(all_indices)
            
            if c_add in code_set:
                continue
                
            # Check validity
            b_rem = c_rem + ball_offsets
            b_add = c_add + ball_offsets
            
            valid = True
            # For x in b_rem, if cnt[x] < 2, x must be in b_add
            add_set = set(b_add.tolist())
            for x in b_rem:
                if cnt[x] < 2 and x not in add_set:
                    valid = False
                    break
                    
            if valid:
                # Perform Swap
                code_list.remove(c_rem)
                code_list.append(c_add)
                code_set.remove(c_rem)
                code_set.add(c_add)
                cnt[b_rem] -= 1
                cnt[b_add] += 1
                
                # Update Best
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
            else:
                # Reject (no state change)
                pass
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.5

    return [idx_to_word(idx) for idx in best_code]
